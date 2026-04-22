from database import Base
from sqlalchemy import Column ,Integer ,String ,DateTime ,Boolean ,Text
from datetime import datetime

# 定义表的格式
class ChatMessages(Base):
    __tablename__ = "chatmessages"

    # 数字类型(int)配合主key，会自动分配数字
    id = Column(Integer ,primary_key=True ,index=True)
    session_id = Column(String(50) ,nullable=False)
    role = Column(String(100) ,nullable=False)
    content = Column(Text ,nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # __tablename__ = 这张表叫什么
    # unique=True = 不能重复
    # index=True = 更方便查找
