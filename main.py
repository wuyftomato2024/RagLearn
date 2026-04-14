from fastapi import FastAPI, UploadFile, File, Form ,HTTPException
from fastapi.responses import JSONResponse
from langchain.memory import ConversationBufferMemory
from utils import ragChat ,normalChat 
import utils
from model import ApiResponse
from typing import List 

# 创建 FastAPI 应用
app = FastAPI()

# 先准备一个最简单的 memory
# 这里只是练习写法，先不用太在意多人共用的问题
memory = ConversationBufferMemory(
    memory_key= "chat_history",
    return_messages=True ,
    output_key="answer" 
)

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

# RAG 问答接口
@app.post("/chat" ,response_model=ApiResponse)
async def ragchat(
    # Form为表单里接收普通数据，File为表单里接收普通数据
    question : str = Form(...),
    openai_api_key : str =Form(...),
    # 把上传文件变成一个List，以上传复数文件
    upload_file : List [UploadFile] | None = File(None),
    top_k :int = Form(3,ge=1,le=3)
):
    
    if upload_file :
        # 调用你已经写好的 utils 里面的函数
        response = await ragChat(
            question = question,
            memory = memory,
            upload_file = upload_file,
            openai_api_key = openai_api_key,
            top_k = top_k)

    elif utils.db is not None :
        response = await ragChat(
            question = question,
            memory = memory,
            upload_file = upload_file,
            openai_api_key = openai_api_key,
            top_k = top_k)
    
    else :
        response = normalChat(
        memory =memory ,
        question = question ,
        openai_api_key = openai_api_key
    )
    # 把结果返回给前端
    return response

# @app.post("/normal_chat")
# def normalModelChat(
#     question : str = Form(...),
#     openai_api_key : str =Form(...),
# ):
#     response = normalChat(
#         memory =memory ,
#         question = question ,
#         openai_api_key = openai_api_key
#     )

#     return response
