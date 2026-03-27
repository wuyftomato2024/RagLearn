from pydantic import BaseModel

class HistoryItem(BaseModel):
    role :str
    content :str

class ChatResponse(BaseModel):
    answer :str
    chatHistory :list[HistoryItem]
    tag: list[str]