from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader , PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import AIMessage ,HumanMessage
from fastapi import HTTPException
from model import ChatResponse ,HistoryItem ,ApiResponse
from prompts import judge_prompt ,chunk_hit_prompt ,summary_prompt ,summary_answer_prompt ,build_qa_prompt
from sqlService import chatCreate ,chatHistoryGet
import os

# *****
# 判断chat模式函数
# *****
def judge(question ,openai_api_key ,memory):
    model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=openai_api_key)
    prompt = judge_prompt()
    
    history_list = chat_history(memory)
    # -4：的意思是，从最后四行开始，到最后 ：的意思是，冒号的左边是从哪里开始，右边是从哪里结束
    result_history = history_list[-4:]

    # format_messages（）是一个对模板专用的添加方法
    message = prompt.format_messages(question=question,history =result_history)

    response = model.invoke(message)
    return response.content

# *****
# RagChat函数
# *****
# 因为fastapi用的是异步上传，所以这里要加上异步“async” 
async def ragChat(question , memory ,upload_file ,openai_api_key ,top_k ,db):
    # 嵌入模型  把每个文本块转成向量（变成数字）
    embedding_model = OpenAIEmbeddings(openai_api_key = openai_api_key)
    # 定义模型
    model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key = openai_api_key)
    # 判断top_k的值，如果非法就报错
    if top_k < 1 or top_k > 3 :
        raise HTTPException(status_code=400 , detail="top_k must be between 1 and 3")

    # 复数写法
    if upload_file :

        docs_list = await handle_upload_files(upload_file)
        # 文档向量化 ，存入数据库 放入(分割好的文件块和嵌入模型，把texts给变成数字)
        db = FAISS.from_documents(docs_list,embedding_model)

    elif db is None :
        raise HTTPException(status_code=400 , detail="please upload a txt or pdf file first")
    
    current_db = db
    
    # 把数据库变成一个“检索器”。后面的search_kwargs是固定写法，是搜索参数的意思，必须是要写成字典的形式
    db_retriever = db.as_retriever(search_kwargs= {"k": top_k}) 
    
    qa_prompt = build_qa_prompt(question)
    # 创建一个“带记忆 + 会检索2资料”的问答链。
    qa = ConversationalRetrievalChain.from_llm(
        llm = model ,
        retriever = db_retriever,
        memory = memory ,
        return_source_documents = True ,
        combine_docs_chain_kwargs = {"prompt":qa_prompt}
    )
    # 调用qa这个问答链，invoke（）里面需要传值的东西，是固定的，不用凭空出现，如果不知道需要用print确认 
    response = qa.invoke({"question" : question})

    # rag内容分支判断（判断这个问题，是否归为总结类）
    summary_kws = ["总结","概括","主要内容","大意","讲了什么"]

    for summary_kw in summary_kws:
        if summary_kw in question :
            summary_response = summary(question ,openai_api_key)
            if summary_response == "True" :
                print("summer success")
                summary_answer_response = summary_answer(openai_api_key ,response)
                result = summary_answer_response               
        else :
            result = response["answer"]

    chunk_hit = chunk_hit_llm(memory ,question ,response ,openai_api_key)

    history_list = chat_history(memory)
    
    source_files = [chunk_hit]
    
    return ApiResponse(
        status = "ok",
        data = ChatResponse(
            answer = result ,
            chatHistory = history_list ,
            tag = source_files)
    ),current_db
   
# *****
# 普通Chat函数
# *****
def normalChat(question ,openai_api_key ,db ,session_id ,):
    model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key =openai_api_key)
    message_list = []

    sql_messages = chatHistoryGet(db = db ,session_id = session_id)

    sql_messages.append(HumanMessage(content = question))
    
    response = model.invoke(sql_messages)

    chatCreate(db = db,session_id =session_id,role = "HumanMessage",content = question)
    chatCreate(db = db,session_id =session_id,role = "AIMessage",content = response.content)

    sql_message_2nd = chatHistoryGet(db = db ,session_id = session_id)

    for sql_message in sql_message_2nd:
        if isinstance(sql_message ,HumanMessage):
            message_list.append(HistoryItem(role = "human",content = sql_message.content))
        if isinstance(sql_message ,AIMessage):
            message_list.append(HistoryItem(role = "ai",content = sql_message.content))

    return ApiResponse(
        status = "ok",
        data = ChatResponse(
            answer = response.content ,
            chatHistory = message_list ,
            tag = [])
    )

# *****
# 上传文件函数
# *****
async def handle_upload_files(upload_file):
    docs_list = []
    # index是自动从0开始赋值，有点类似于i等于0，i++那种感觉，而enumerate()是一个特殊用法，用来给东西增加编号
    # enumerate()和index是绑定用法，其实index本身是没有值的，这个值反而是enumerate给赋予的
    for index , file in enumerate(upload_file):
        file_name = file.filename
        file_content = await file.read()
        
        if not file_content :
            raise HTTPException(status_code=400 ,detail=f"{file_name} is empty")
        
        if file_name.endswith(".txt"):
            temp_file_path = f"temp{index}.txt"
            with open (temp_file_path,"wb") as temp_file :
                temp_file.write(file_content)
            loader = TextLoader(temp_file_path ,encoding="utf-8")
            docs = loader.load()
            
        elif file_name.endswith(".pdf"):
            temp_file_path = f"temp{index}.pdf"
            with open (temp_file_path,"wb") as temp_file :
                temp_file.write(file_content)
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

        else :
            raise HTTPException(status_code=400 ,detail=f"{file_name} is not a supported file type. Only txt and pdf are supported. ")

        # 删除生成的本地文件
        os.remove(temp_file_path)

        text_splitters = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50,
                separators=["\n", "。", "!", "？", ",", "、", ""]
                )
        texts = text_splitters.split_documents(docs)

        # 循环 给text里面a新增一个metadata，名字叫做file_name，而内容是上面的file_name
        for text in texts:
            text.metadata["file_name"] = file_name

        # 用extend的原因是，用append的话会整一个list塞进去docs_list里面去，而extend是把docs里面的内容全部拆开，再放到extend里面去
        docs_list.extend(texts)
    
    return docs_list

# *****
# 整理history
# *****
def chat_history(memory):
    # 先创建history列表的空值
    history_list = []
    # for循环
    for msg in memory.chat_memory.messages :
        # isinstance()为判断的用法，判断（a等于b）
        if isinstance(msg,HumanMessage):
            # 如果a等于b的话就往history_list里面添加以下字典
            history_list.append(HistoryItem(role = "human",content = msg.content))
        elif isinstance(msg,AIMessage):
            history_list.append(HistoryItem(role = "ai" , content = msg.content))
    
    return history_list

# *****
# 整理 回答的问题，整理成一个大文件块
# *****
def chunk_hit(response):

    ai_text_map = []

    for doc in response["source_documents"]:
        ai_text = f"filename : {doc.metadata['file_name']} \n file_content :{doc.page_content}"
        ai_text_map.append(ai_text)

    ai_text_all = "\n".join(ai_text_map)

    return ai_text_all

# *****
# 命中文件来源llm
# *****
def chunk_hit_llm(memory ,question ,response ,openai_api_key):
    ai_text_all = chunk_hit(response)

    model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=openai_api_key)
    prompt = chunk_hit_prompt()
    
    history_list = chat_history(memory)

    result_history = history_list[-4:]

    # format_messages（）是一个对模板专用的添加方法
    message = prompt.format_messages(
        question= question ,
        history = result_history ,
        ai_text = ai_text_all
        )

    chunk_response = model.invoke(message)
    return chunk_response.content

# *****
# 判断「概括/总结整份上传文件的整体内容」llm
# *****
def summary(question ,openai_api_key):
    model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key =openai_api_key)

    prompt = summary_prompt()

    message = prompt.format_messages(question = question)

    summary_response = model.invoke(message)

    return summary_response.content

# *****
# 提取response结果
# *****
def summary_texts(response):
    summary_text_map = []
    for doc in  response["source_documents"]:
        answer = doc.page_content
        summary_text_map.append(answer)

    summary_text = "\n".join(summary_text_map)

    return summary_text

# *****
# 判断「结果生成回答」llm
# *****
def summary_answer(openai_api_key ,response):
    text = summary_texts(response)

    model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key =openai_api_key)

    prompt = summary_answer_prompt()

    message = prompt.format_messages(result = text)

    summary_answer_response = model.invoke(message)

    return summary_answer_response.content