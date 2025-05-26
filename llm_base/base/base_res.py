class ResponseScoreReason(object):
    def __init__(self, state: bool = False, response_json: dict = None):
        """
        Initialize the ResponseScoreReason object.
        """
        super().__init__()
        self.state: bool = state
        self.response_json: dict = response_json if response_json is not None else {}

    # score: float
    # reason: str = ""

#     class Config:
#         extra = "forbid"
#         validate_assignment = True


# class ResponseNameReason(BaseModel):
#     name: str
#     reason: str = ""

#     class Config:
#         extra = "forbid"
#         validate_assignment = True


# class ResponseScoreTypeNameReason(BaseModel):
#     score: int
#     type: str = "Type"
#     name: str = "Name"
#     reason: str = ""

#     class Config:
#         extra = "forbid"
#         validate_assignment = True
