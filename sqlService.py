from db_model import DBResponse
from db_format import ChatMessages
from fastapi import HTTPException
from langchain_core.messages import AIMessage ,HumanMessage

def chatCreate(sql_db , session_id ,role , content):

    newChat = ChatMessages(
        session_id = session_id ,
        role =  role ,
        content = content
    )

    sql_db.add(newChat)
    sql_db.commit()

    return DBResponse(
        status = "ok",
        data={
            "created":True ,
            "data":{
                newChat.session_id ,
                newChat.role ,
                newChat.content
            }
        }

    )

def chatHistoryGet(sql_db ,session_id):
    chatHistory_map = []

    chatHistorys = sql_db.query(ChatMessages).filter(ChatMessages.session_id == session_id).all()

    if chatHistorys is None:
        raise HTTPException(status_code= 400 ,detail="chatHistory is None")
    
    for chatHistory in chatHistorys :
        if chatHistory.role == "HumanMessage" :
            chatHistory_map.append(HumanMessage(content = chatHistory.content))
        if chatHistory.role == "AIMessage" : 
            chatHistory_map.append(AIMessage(content = chatHistory.content))

    return chatHistory_map

def chatDelete(sql_db ,session_id):
    chatHistorys = sql_db.query(ChatMessages).filter(ChatMessages.session_id == session_id).all()

    if not chatHistorys :
        raise HTTPException(status_code=404 ,detail="chatMessage in None")

    for chatHistory in chatHistorys :
        sql_db.delete(chatHistory)
    sql_db.commit()

    return DBResponse(
        status = "ok",
        data={
            "Deleted":True ,
            "data":{
                None
            }
        }

    )
  

def chatMessages(user):
    return {
        "id": user.id,
        "session_id": user.session_id,
        "role": user.role,
        "content":user.content,
        "created_at":user.created_at  
    }