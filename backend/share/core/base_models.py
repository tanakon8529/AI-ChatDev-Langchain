
from pydantic import BaseModel

class SendKafka(BaseModel):
    topic: str
    data: dict