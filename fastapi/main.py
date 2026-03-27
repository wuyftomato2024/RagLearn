from fastapi import FastAPI, UploadFile, File, Form
from langchain.memory import ConversationBufferMemory
from utils_txt import ragChat
import uvicorn

# 创建 FastAPI 应用
app = FastAPI()

# 先准备一个最简单的 memory
# 这里只是练习写法，先不用太在意多人共用的问题
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 测试接口
@app.get("/")
def root():
    return {"message": "FastAPI is running"}

# RAG 问答接口
@app.post("/chat")
def chat_with_txt(
    question: str = Form(...),         # 前端传来的问题
    openai_api_key: str = Form(...),   # 前端传来的 API Key
    upload_file: UploadFile = File(...) # 前端上传的 txt 文件
):
    # 调用你已经写好的 utils 里面的函数
    response = ragChat(
        question=question,
        memory=memory,
        upload_file=upload_file.file,   # 注意：这里传的是底层文件对象
        openai_apk_key=openai_api_key
    )

    # 把结果返回给前端
    return {
        "answer": response["answer"],
        "chat_history": str(response["chat_history"])
    }

if __name__ == "__main__" :
    uvicorn.run("main:app" , reload=True)