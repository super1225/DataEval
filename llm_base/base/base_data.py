from typing import Dict, List, Optional

from pydantic import BaseModel


class MetaData(BaseModel):
    """
    Metadata, output of converter.
    """
    data_id: str
    prompt: str = None
    content: str = None
    image: Optional[List] = None
    raw_data: Dict = {}
