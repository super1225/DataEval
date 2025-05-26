
import os
import time
import json
import enum
import functools

import torch
import datasets
import bert_score
import numpy as np
import transformers
from typing import List, Dict
from swift.llm import sft_main, TrainArguments
from swift.plugin.metric import METRIC_MAPPING, MeanMetric, preprocess_logits_for_acc

from .rule import BaseRule, DynamicRuleConfig, ModelRes
from quality_filter.util.files import input_data_to_str
from quality_filter.iterator.metrics.qa_eval import evaluate_qa


class ConsistencyMetric(str, enum.Enum):
    Perplexity = "Perplexity"


class ConsistencyProcess(BaseRule):
    """check consistency"""
    def __init__(self, model_name: str, metric: ConsistencyMetric | str):
        super().__init__()
        self.process_name = "consistency_process"
        self.model_name = model_name
        self.metric = metric if isinstance(metric, ConsistencyMetric) else ConsistencyMetric(metric)
        self.model_device = "cuda" if torch.cuda.is_available() else "cpu"
        
    
    def __compute_perplexity(self, text: str, tokenizer: transformers.PreTrainedTokenizerBase, model: transformers.PreTrainedModel) -> float:
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.cuda()

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = torch.exp(loss)
        return perplexity.item()


    def __process__(self, input_data) -> ModelRes:
        input_data = input_data_to_str(input_data)
        
        if self.metric in [ConsistencyMetric.Perplexity]:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name).to(self.model_device)
            model.eval()
            
            # 困惑度评分
            ppl = self.__compute_perplexity(input_data, tokenizer, model)
            
            return ModelRes(
                error_status=False if ppl < 100 else True,
                type=self.metric.value,
                name=self.__class__.__name__,
                value=ppl,
                reason=[f"Perplexity: {ppl}"]
            )
            
            
class CompletionMetric(str, enum.Enum):
    BERT_SCORE = "BERT_SCORE"
    
    
class CompletionProcess(BaseRule):
    """check completion"""
    def __init__(self, refs_data_path: str, model_path: str, num_layers: int, metric: CompletionMetric | str) -> None:
        super().__init__()
        self.process_name = "completion_process"
        self.refs_data_path = refs_data_path
        self.model_path = model_path
        self.num_layers = num_layers
        self.metric = metric if isinstance(metric, CompletionMetric) else CompletionMetric(metric)
        
        with open(self.refs_data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            self.refs_data = [item.get('text', item) for item in data]
        
        
    def __process__(self, input_data: str | List[str] | Dict[str | int, str]) -> ModelRes:
        input_data = input_data_to_str(input_data)
        
        if self.metric in [CompletionMetric.BERT_SCORE]:
            P, R, F1 = bert_score.score([input_data], self.refs_data, model_type=self.model_path, lang="en", rescale_with_baseline=True, num_layers=self.num_layers, verbose=False)
            
            return ModelRes(
                error_status=False,
                type=self.metric.value,
                name=self.__class__.__name__,
                reason=[f"P: {P.mean().item():.4f}, R: {R.mean().item():.4f}, F1: {F1.mean().item():.4f}"]
            )
            
            
class AdaptabilityTaskType(str, enum.Enum):
    QA = "QA"
    SUMMARIZATION = "SUMMARIZATION"
    

class AdaptabilityProcess(BaseRule):
    """check adaptability"""
    def __init__(
        self, 
        model_name: str, 
        task_type: AdaptabilityTaskType | str, 
        select_ratio: float = 0.01, 
        train_type: str = "lora", 
        epochs: int = 1,
        seed: int = 1337
        ) -> None:
        
        super().__init__()
        self.process_name = "adaptability_process"
        self.model_name = model_name
        self.select_ratio = select_ratio
        self.task_type = task_type if isinstance(task_type, AdaptabilityTaskType) else AdaptabilityTaskType(task_type)
        self.train_type = train_type
        self.epochs = epochs
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.global_results: List[Dict[str, float]] = []
        os.makedirs("./cache", exist_ok=True)
        transformers.set_seed(1337)
        
        
    def __gen_prompt(self, task_type: AdaptabilityTaskType, item: Dict[str, str]) -> str:
        if task_type == AdaptabilityTaskType.QA:
            prompt = "Given Context:\n" + item["context"] + "\n\nAnswer the question:\n" + item['question']
        elif task_type == AdaptabilityTaskType.SUMMARIZATION:
            prompt = "Summarize the following text:\n" + item['context']
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        return prompt
    
        
    def __preprocess_dataset(self, input_data_file_path: str, select_ratio: float, task_type: AdaptabilityTaskType, seed: int = 1337) -> str:
        # dataset = datasets.load_dataset("json", data_files=input_data_file_path)["train"]
        dataset = datasets.Dataset.from_list(input_data_file_path, split="train")
        if select_ratio < 1.0:
            sample_size = int(len(dataset) * select_ratio)
            sampled_dataset = dataset.shuffle(seed=seed).select(range(sample_size))
        
        output_file = f"./cache/{task_type}_sampled_{select_ratio}_{int(time.time())}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in sampled_dataset:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": self.__gen_prompt(task_type, item)},
                    {"role": "assistant", "content": item["answer"]}
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        return output_file


    def __compute_nlg_metrics(self, tokenizer: transformers.PreTrainedTokenizerBase, prediction: transformers.trainer_utils.EvalPrediction) -> Dict[str, float]:
        preds, labels = prediction[0], prediction[1]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        import jieba
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from rouge.rouge import Rouge
        
        score_dict = {key: MeanMetric() for key in ['rouge-1', 'rouge-2', 'rouge-l', 'bleu-4']}

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            if not hypothesis or not reference:
                continue
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))[0]
            for k, v in scores.items():
                score_dict[k].update(v['f'])
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict['bleu-4'].update(bleu_score)
        
        results = {k: round(v.compute()['value'] * 100, 6) for k, v in score_dict.items()}
        self.global_results.append(results)
        return results
        

    def __compute_em_metrics(self, tokenizer: transformers.PreTrainedTokenizerBase, prediction: transformers.trainer_utils.EvalPrediction) -> Dict[str, float]:
        preds, labels = prediction[0], prediction[1]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds_dict = {idx: pred for idx, pred in enumerate(decoded_preds)}
        labels_dict = {idx: [label] for idx, label in enumerate(decoded_labels)}

        results = evaluate_qa(preds_dict, labels_dict)
        self.global_results.append(results)
        return results
        
        
    def __process__(self, input_data: str | List[str] | Dict[str | int, str]) -> ModelRes:
        # 数据预处理
        preprocessed_dataset_path = self.__preprocess_dataset(
            input_data_file_path=input_data,
            select_ratio=self.select_ratio,
            task_type=self.task_type,
            seed=self.seed
        )
        # 评估指标预处理
        curr_tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        METRIC_MAPPING.update({
            "em": (functools.partial(self.__compute_em_metrics, curr_tokenizer), preprocess_logits_for_acc),
            "nlg": (functools.partial(self.__compute_nlg_metrics, curr_tokenizer), preprocess_logits_for_acc)
        })

        # 选定指标
        if self.task_type in [AdaptabilityTaskType.QA]:
            self.metrics = "em"
        elif self.task_type in [AdaptabilityTaskType.SUMMARIZATION]:
            self.metrics = "nlg"
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # 训练并评估
        result = sft_main(TrainArguments(
            model=self.model_name,
            train_type=self.train_type,
            dataset=[preprocessed_dataset_path],
            torch_dtype='bfloat16',
            stream=True,
            check_model=False,
            torch_empty_cache_steps=1,
            save_total_limit=0,
            logging_dir="./cache/logs",
            output_dir="./cache/output",
            
            eval_on_start=True,
            eval_steps=100,
            logging_steps=10,
            max_epochs=self.epochs,
            
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            max_length=512,
            max_new_tokens=128,
            warmup_ratio=0.05,
            
            metric=self.metrics
        ))
        
        # 指标取最大值
        final_max_metrics_dict = {}
        if self.global_results:
            for result_dict in self.global_results:
                for metric_key, metric_value in result_dict.items():
                    if metric_key not in final_max_metrics_dict or metric_value > final_max_metrics_dict[metric_key]:
                        final_max_metrics_dict[metric_key] = metric_value
            
            
        return ModelRes(
            error_status=False,
            type=self.task_type.value,
            name=self.__class__.__name__,
            reason=[f"Max {metric}: {value}" for metric, value in final_max_metrics_dict.items()]
        )