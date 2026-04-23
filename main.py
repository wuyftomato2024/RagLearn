from fastapi import FastAPI, UploadFile, File, Form ,HTTPException ,Depends
from fastapi.responses import JSONResponse
from langchain.memory import ConversationBufferMemory
from utils import ragChat ,normalChat ,judge 
from sqlService import chatCreate ,chatHistoryGet
from model import ApiResponse
from typing import List 
from database import engine ,Base ,SessionLocal
import os

# 创建 FastAPI 应用
app = FastAPI()

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try :
        yield db
    finally :
        db.close()

memory_map = {}
db_map = {}

# 测试接口
@app.get("/")
def root():
    return {"message":"Fastapi was running"}

# HTTPException之所以好像没有调用，是因为后端触发了raise的话，fastapi其实就已经知道有这个东西的存在了，只不过没写出来的话，就是以别的格式返回现在做的是让他以一个固定格式返回 。exc :HTTPException 这种写法是给他加一个类型说明，让人一眼能看出来这个是哪种异常
@app.exception_handler(HTTPException)
async def setError(request,exc :HTTPException):
    return JSONResponse(
        status_code = exc.status_code,
        content = {
            "status":"fail",
            "data" : None,
            "detail":exc.detail
        }
    )

# Exception是意料外的错误
@app.exception_handler(Exception)
async def error(request,exc :Exception):
    return JSONResponse(
        status_code= 500,
        content={
            "status":"fail",
            "data" : None,
            "detail":str(exc)
        }
    )

# *****
# ragChat接口
# *****
@app.post("/chat" ,response_model=ApiResponse)
async def ragchat(
    # Form为表单里接收普通数据，File为表单里接收普通数据
    question : str = Form(...),
    openai_api_key : str =Form(...),
    # 把上传文件变成一个List，以上传复数文件
    upload_file : List [UploadFile] | None = File(None),
    top_k :int = Form(3,ge=1,le=3),
    session_id :int = Form(...) ,
    sql_db = Depends(get_db)
):
    # if session_id not in db_map :
    #     vector_db = None
    #     db_map[session_id] = vector_db

    # current_db = db_map[session_id]    

    vector_db_path = f"faiss_db/{session_id}/"
    vector_db_flag = os.path.exists(vector_db_path)

    if session_id not in memory_map :
        memory = ConversationBufferMemory(
            memory_key= "chat_history",
            return_messages=True ,
            output_key="answer"
            )
        memory_map[session_id] = memory
    current_memory = memory_map[session_id]

    if upload_file :
        # 调用你已经写好的 utils 里面的函数
        response = await ragChat(
            question = question,
            memory = current_memory,
            upload_file = upload_file,
            openai_api_key = openai_api_key,
            top_k = top_k ,
            sql_db = sql_db ,
            session_id = session_id
            )
    
    elif not upload_file and vector_db_flag :

        judge_flag = True

        rag_kws = ["文件","文档","pdf","上传","总结","概括"]
        for rag_kw in rag_kws :
            if rag_kw in question :
                response = await ragChat(
                question = question,
                memory = current_memory,
                upload_file = upload_file,
                openai_api_key = openai_api_key,
                top_k = top_k ,
                sql_db = sql_db ,
                session_id = session_id
                )
                print("is real rag")
                judge_flag = False
                break
            
        history_kws = ["上一个问题" ,"刚刚说的" ,"继续刚才" ,"刚才那个" ,"上一条"]
        for history_kw in history_kws :
            if history_kw in question :
                response = normalChat(
                # memory =current_memory ,
                question = question ,
                openai_api_key = openai_api_key ,
                sql_db = sql_db ,
                session_id = session_id
                )
                print("is real normal")
                judge_flag = False
                break

        if judge_flag:

            judge_response = judge(
                question = question,
                openai_api_key = openai_api_key ,
                memory = current_memory
            ).strip().lower()

            if judge_response == "rag" :
                response = await ragChat(
                question = question,
                memory = current_memory,
                upload_file = upload_file,
                openai_api_key = openai_api_key,
                top_k = top_k ,
                sql_db = sql_db ,
                session_id = session_id
                )

                print("db and rag success")

            elif judge_response == "history":
                response = normalChat(
                question = question ,
                openai_api_key = openai_api_key ,
                sql_db = sql_db ,
                session_id = session_id
                )
                print("db and history success")

            elif judge_response == "normal":
                response = normalChat(
                question = question ,
                openai_api_key = openai_api_key ,
                sql_db = sql_db ,
                session_id = session_id
                )
                print("db and normal success")

            else :
                response = normalChat(
                question = question ,
                openai_api_key = openai_api_key ,
                sql_db = sql_db ,
                session_id = session_id
                )
                print("db and normal success")

    else :
        response = normalChat(
        question = question ,
        openai_api_key = openai_api_key ,
        sql_db = sql_db ,
        session_id = session_id
        )
        # print(memory_map)
        # print(db_map)
    # 把结果返回给前端
    return response

# @app.post("/chat/db")
# def createChat(body : DBCreate ,db = Depends(get_db)):
#     response = chatCreate(
#         db = db ,
#         content = body.content ,
#         role = body.role ,
#         session_id = body.session_id
#         )

#     return  response

# @app.get("/chat/get")
# def getHistoryChat(session_id :int,db = Depends(get_db)):
#     response = chatHistoryGet(
#         session_id = session_id ,
#         db = db
#     )
#     return response