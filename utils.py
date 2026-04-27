from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader , PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage ,HumanMessage ,SystemMessage
from fastapi import HTTPException
from model import ChatResponse ,HistoryItem ,ApiResponse
from prompts import judge_prompt ,chunk_hit_prompt ,summary_prompt ,summary_answer_prompt ,defult_normalChat ,simple_normalChat
from sqlService import chatCreate ,chatHistoryGet
import os
import shutil
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# *****
# 判断chat模式函数
# *****
def judge(question ,openai_api_key ,sql_db ,session_id ,model_flag):
    ai_model ,_ =ai_model_select(model_flag ,openai_api_key)
    model = ai_model
    prompt = judge_prompt()
    
    history_list = chatHistoryGet(sql_db = sql_db,session_id = session_id)
    # -4：的意思是，从最后四行开始，到最后 ：的意思是，冒号的左边是从哪里开始，右边是从哪里结束
    result_history = history_list[-4:]

    # format_messages（）是一个对模板专用的添加方法
    message = prompt.format_messages(question=question,history =result_history)

    response = model.invoke(message)
    return response.content

# *****
# RagChat函数
# *****
async def ragChat(question , upload_file ,openai_api_key ,top_k ,sql_db ,session_id ,model_flag):
    ai_model ,ai_embedding =ai_model_select(model_flag ,openai_api_key)

    # 嵌入模型  把每个文本块转成向量（变成数字）
    embedding_model = ai_embedding
    # embedding_model = OllamaEmbeddings(model="bge-large")

    # 定义模型
    model = ai_model
    # model = ChatOllama(model="deepseek-r1:14b")

    # 判断top_k的值，如果非法就报错
    if top_k < 1 or top_k > 3 :
        raise HTTPException(status_code=400 , detail="top_k must be between 1 and 3")
    
    if upload_file :
        docs_list = await handle_upload_files(upload_file)
        # 文档向量化 ，存入数据库 放入(分割好的文件块和嵌入模型，把texts给变成数字)
        vector_db = FAISS.from_documents(docs_list,embedding_model)
        save_local_vector_db(session_id = session_id,vector_db =vector_db)       
            
    else :
        local_vector_db = load_local_vector_db(session_id = session_id ,embedding_model = embedding_model)
        if not local_vector_db:
            raise HTTPException(status_code=400 , detail="please upload a txt or pdf file first")
        vector_db = local_vector_db

    # 向量库检索
    chunk_content_text ,chunk_texts= chunk_context(vector_db ,top_k ,question)
    
    qa_prompt = answer_model(question)

    sql_messages = chatHistoryGet(sql_db = sql_db ,session_id = session_id)

    human_text = f"Context:\n{chunk_content_text}\n\nQuestion:\n{question}"

    message = [SystemMessage(content = qa_prompt)] + sql_messages + [HumanMessage(content =human_text)]

    response = model.invoke(message)

    result = response.content

    # rag内容分支判断（判断这个问题，是否归为总结类）
    summary_kws = ["总结","概括","主要内容","大意","讲了什么"]
    for summary_kw in summary_kws:
        if summary_kw in question :
            summary_response = summary(question ,openai_api_key ,model_flag)
            if summary_response == "True" :
                print("summary success")
                summary_answer_response = summary_answer(openai_api_key ,vector_db ,top_k ,question ,model_flag)
                result = summary_answer_response

    chatCreate(sql_db =sql_db, session_id =session_id,role = "HumanMessage" , content = question)
    chatCreate(sql_db =sql_db, session_id =session_id,role = "AIMessage" , content = result)

    chunk_hit = chunk_hit_llm(question ,chunk_texts ,sql_db ,session_id ,openai_api_key ,model_flag)

    message_list = sql_message_process(sql_db =sql_db, session_id =session_id)
    
    source_files = [chunk_hit]
    
    return ApiResponse(
        status = "ok",
        data = ChatResponse(
            answer = result ,
            chatHistory = message_list ,
            tag = source_files)
    )
   
# *****
# 普通Chat函数
# *****
def normalChat(question ,openai_api_key ,sql_db ,session_id ,model_flag):
    ai_model ,_ =ai_model_select(model_flag ,openai_api_key)
    model = ai_model

    qa_prompt = answer_model(question)

    sql_messages = chatHistoryGet(sql_db = sql_db ,session_id = session_id)
    messages = [SystemMessage(content = qa_prompt)] + sql_messages + [HumanMessage(content = question)]
   
    response = model.invoke(messages)

    chatCreate(sql_db = sql_db,session_id =session_id,role = "HumanMessage",content = question)
    chatCreate(sql_db = sql_db,session_id =session_id,role = "AIMessage",content = response.content)

    message_list = sql_message_process(sql_db =sql_db, session_id =session_id)

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
# 整理 回答的问题，整理成一个大文件块
# *****
def chunk_hit(chunk_texts):
    ai_text_map = [f"filename : {doc.metadata['file_name']} \n file_content :{doc.page_content}" for doc in chunk_texts]

    ai_text_all = "\n".join(ai_text_map)

    return ai_text_all

# *****
# 命中文件来源llm
# *****
def chunk_hit_llm(question ,chunk_texts ,sql_db ,session_id ,openai_api_key ,model_flag):
    ai_text_all = chunk_hit(chunk_texts)

    ai_model ,_ =ai_model_select(model_flag ,openai_api_key)
    model = ai_model

    prompt = chunk_hit_prompt()
    
    history_list = chatHistoryGet(sql_db = sql_db,session_id = session_id)

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
# 判断此次问题是否需要summary
# *****
def summary(question ,openai_api_key ,model_flag):
    ai_model ,_ =ai_model_select(model_flag ,openai_api_key)
    model = ai_model

    prompt = summary_prompt()

    message = prompt.format_messages(question = question)

    summary_response = model.invoke(message)

    return summary_response.content

# *****
# 向量库检索 ，提取chunk结果
# *****
def chunk_context(vector_db ,top_k ,question):
    # 把数据库变成一个“检索器”。后面的search_kwargs是固定写法，是搜索参数的意思，必须是要写成字典的形式
    db_retriever = vector_db.as_retriever(search_kwargs= {"k": top_k}) 

    # chunk context定义
    chunk_texts = db_retriever.invoke(question)
    chunk_map = [chunk_text.page_content for chunk_text in chunk_texts ]
    chunk_content_text =  "\n".join(chunk_map)

    return chunk_content_text ,chunk_texts

# *****
# 通过搜索到的chunk生成回答问题
# *****
def summary_answer(openai_api_key ,vector_db ,top_k ,question ,model_flag):
    text ,_ = chunk_context(vector_db ,top_k ,question)

    ai_model ,_ =ai_model_select(model_flag ,openai_api_key)
    model = ai_model

    prompt = summary_answer_prompt()

    message = prompt.format_messages(result = text)

    summary_answer_response = model.invoke(message)

    return summary_answer_response.content

# *****
# 向量库保存到本地函数
# *****
def save_local_vector_db(session_id ,vector_db):
    vector_db_path = f"faiss_db/{session_id}/"
    vector_db.save_local(vector_db_path)

# *****
# 向量库读取本地函数
# *****
def load_local_vector_db(session_id ,embedding_model):
    vector_db_path = f"faiss_db/{session_id}/"
    # 因为保存到本地的向量库，不会把嵌入模型也保存到本地，所以读取的时候必须先把向量库给重新附加一次
    vector_db = FAISS.load_local(vector_db_path ,embedding_model ,allow_dangerous_deserialization=True)
    if vector_db is None :
        raise HTTPException(status_code=400 ,detail="Not local db")

    return vector_db

# *****
# 按照session删除内容
# *****
def delete_vector_db(session_id) :
    vector_db_path = f"faiss_db/{session_id}/"
    vector_db_flag = os.path.exists(vector_db_path)
    if vector_db_flag == False:
        raise HTTPException(status_code=404 ,detail="the file is not")
    shutil.rmtree(vector_db_path)

# *****
# 前端输出转换
# *****    
def sql_message_process(sql_db ,session_id):
    sql_messages = chatHistoryGet(sql_db = sql_db,session_id = session_id)

    message_list = []

    for sql_message in sql_messages:
        if isinstance(sql_message ,HumanMessage):
            message_list.append(HistoryItem(role = "human",content = sql_message.content))
        if isinstance(sql_message ,AIMessage):
            message_list.append(HistoryItem(role = "ai",content = sql_message.content))

    return message_list

# *****
# 回复模式判断
# *****  
def answer_model(question):
    simple_words = ["简单","简洁","简短","少废话","易懂","少例子"]
    is_simple_mode = False
    for kw in simple_words:
        if kw in question :
            is_simple_mode = True
            break
    if is_simple_mode is True :
        qa_prompt = simple_normalChat()

    else:
        qa_prompt = defult_normalChat()

    return qa_prompt 


# *****
# ai模型选择
# *****  
def ai_model_select(model_flag ,openai_api_key):
    if model_flag == "openai" :
        ai_model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key =openai_api_key)
        embedding_model = OpenAIEmbeddings(openai_api_key = openai_api_key)
    elif model_flag == "ollama" :
        ai_model = ChatOllama(model="deepseek-r1:14b")
        embedding_model = OllamaEmbeddings(model="bge-large")
    else :
        raise HTTPException(status_code=500 ,detail="modle is miss")
    
    return ai_model ,embedding_model