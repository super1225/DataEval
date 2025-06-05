from quality_filter.loader.base import DataProvider
from typing import Optional, Union, Sequence, Mapping, Dict, Any
from torch.utils.data import DataLoader
import datasets

class HuggingFaceSource(DataProvider):
    """Represents the source of a Hugging Face dataset used in Dingo Tracking."""

    def __init__(
            self,
            path: Optional[str] = None,
            split: Optional[str] = None,
            config_name: Optional[str] = None,
            data_dir: Optional[str] = None,
            data_files: Optional[
                Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
            ] = None,
            batch_size: Optional[int] = None,
            revision: Optional[Union[str, datasets.Version]] = None,
            trust_remote_code: Optional[bool] = None,
    ):
        """Create a `HuggingFaceSource` instance.
        Arguments in `__init__` match arguments of the same name in
        [`datasets.load_dataset()`](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/loading_methods#datasets.load_dataset).
        The only exception is `config_name` matches `name` in `datasets.load_dataset()`, because
        we need to differentiate from `dingo.data.Dataset` `name` attribute.
        Args:
            input_args: A `InputArgs` instance to load the dataset from.
            config_name: The name of the Hugging Face dataset configuration.
            data_dir: The `data_dir` of the Hugging Face dataset configuration.
            data_files: Paths to source data file(s) for the Hugging Face dataset configuration.
            revision: Version of the dataset script to load.
            trust_remote_code: Whether to trust remote code from the dataset repo.
        """
        self.path = path
        self.config_name = config_name
        self.data_dir = data_dir
        self.data_files = data_files
        self.batch_size = batch_size if batch_size is not None else 32
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        if split != None:
            self.split = split
        else:
            self.split = 'train'
        super().__init__()

    @staticmethod
    def get_source_type() -> str:
        return "hugging_face"

    def load(self, **kwargs) -> datasets.Dataset:
        """Load the Hugging Face dataset based on `HuggingFaceSource`.
        Args:
            kwargs: Additional keyword arguments used for loading the dataset with the Hugging Face
                `datasets.load_dataset()` method.
        Returns:
            An instance of `datasets.Dataset`.
        """
        import datasets
        from packaging.version import Version
        load_kwargs = {
            "path": self.path,
            "name": self.config_name,
            "data_dir": self.data_dir,
            "data_files": self.data_files,
            "split": self.split,
            "revision": self.revision,
        }
        # this argument only exists in >= 2.16.0
        if Version(datasets.__version__) >= Version("2.16.0"):
            load_kwargs["trust_remote_code"] = self.trust_remote_code
        intersecting_keys = set(load_kwargs.keys()) & set(kwargs.keys())
        if intersecting_keys:
            raise KeyError(
                f"Found duplicated arguments in `HuggingFaceSource` and "
                f"`kwargs`: {intersecting_keys}. Please remove them from `kwargs`."
            )
        return datasets.load_dataset(**load_kwargs)

    def to_dict(self) -> Dict[Any, Any]:
        return {
            "path": self.path,
            "config_name": self.config_name,
            "data_dir": self.data_dir,
            "data_files": self.data_files,
            "split": str(self.split),
            "revision": self.revision,
        }
    
    def iter(self):
        split_ds = self.load()
        train_loader = DataLoader(split_ds, batch_size=self.batch_size, shuffle=True)
        for batch in train_loader:
            yield batch
