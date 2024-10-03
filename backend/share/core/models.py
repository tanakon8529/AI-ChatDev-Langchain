
from pydantic import BaseModel

class DynamicBaseModel(BaseModel):
    class Config:
        extra = 'allow'