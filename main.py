from fastapi import FastAPI, UploadFile, File, Form
from langchain.memory import ConversationBufferMemory
from utils_txt import ragChat
from model import ChatResponse
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

# RAG 问答接口
@app.post("/chat" ,response_model=ChatResponse)
async def ragchat(
    # Form为表单里接收普通数据，File为表单里接收普通数据
    question : str = Form(...),
    openai_api_key : str =Form(...),
    # 把上传文件变成一个List，以上传复数文件
    upload_file : List [UploadFile] | None = File(None),
    top_k :int = Form(2,ge=1,le=3)

    
):

    # 调用你已经写好的 utils 里面的函数
    response = await ragChat(
        question = question,
        memory = memory,
        upload_file = upload_file,
        openai_api_key = openai_api_key,
        top_k = top_k)

    # 把结果返回给前端
    return response
