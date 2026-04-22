from pydantic import BaseModel
from typing import Any

class DBResponse(BaseModel):
    status :str
    data :Any

class DBCreate(BaseModel):
    session_id :int 
    role : str
    content : str