from db_model import DBResponse
from db_format import ChatMessages
from fastapi import HTTPException
from langchain_core.messages import AIMessage ,HumanMessage

def chatCreate(db , session_id ,role , content):

    newChat = ChatMessages(
        session_id = session_id ,
        role =  role ,
        content = content
    )

    db.add(newChat)
    db.commit()

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

def chatHistoryGet(db ,session_id):
    chatHistory_map = []

    chatHistorys = db.query(ChatMessages).filter(ChatMessages.session_id == session_id).all()

    if chatHistorys is None:
        raise HTTPException(status_code= 400 ,detail="chatHistory is None")
    
    for chatHistory in chatHistorys :
        if chatHistory.role == "HumanMessage" :
            chatHistory_map.append(HumanMessage(content = chatHistory.content))
        if chatHistory.role == "AIMessage" : 
            chatHistory_map.append(AIMessage(content = chatHistory.content))

    return chatHistory_map
  

def chatMessages(user):
    return {
        "id": user.id,
        "session_id": user.session_id,
        "role": user.role,
        "content":user.content,
        "created_at":user.created_at  
    }