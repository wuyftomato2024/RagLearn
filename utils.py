from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader , PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import AIMessage ,HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from fastapi import HTTPException
from model import ChatResponse ,HistoryItem ,ApiResponse
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

    history_list = chat_history(memory)
    
    source_files = source_file_organize(response)
    
    return ApiResponse(
        status = "ok",
        data = ChatResponse(
            answer = response["answer"] ,
            chatHistory = history_list ,
            tag = source_files)
    ),current_db
   
# *****
# 普通Chat函数
# *****
def normalChat(question ,memory ,openai_api_key):
    model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key =openai_api_key)

    response = model.invoke(question)
    # print(response)

    memory.chat_memory.messages.append(HumanMessage(content=question))
    memory.chat_memory.messages.append(AIMessage(content=response.content))

    history_list = chat_history(memory)

    return ApiResponse(
        status = "ok",
        data = ChatResponse(
            answer = response.content ,
            chatHistory = history_list ,
            tag = [])
    )

# *****
# 定义回复模式提示词模板函数
# *****
def build_qa_prompt(question):
# 定义模型回答模板  用""""内容""""的写法是因为，写多行
    simple_system_message = """Please answer concisely and clearly.
                                Only answer the main point.
                                Do not give too many examples.
                                If the answer is not clearly stated in the provided context, do not make up information.
                                Instead, say that the context does not clearly mention it."""
    normal_system_message = """ 清楚回答问题
                                可适当说明
                                没有资料就明确说没有
                                不要编造"""
    human_message = """Context:{context}
                        Question:{question}"""
    
    # 创建关键词库
    simple_words = ["简单","简洁","简短","少废话","易懂","少例子"]
    # 简单模式的开关
    is_simple_mode = False
    for kw in simple_words:
        if kw in question :
            is_simple_mode = True
            break
    if is_simple_mode is True :
        qa_prompt = ChatPromptTemplate.from_messages([
            (
                "system", simple_system_message
            ),
            (
                "human", human_message
            )
        ])
    
    else:
        qa_prompt = ChatPromptTemplate.from_messages([
            (
                "system", normal_system_message
            ),
            (
                "human", human_message
            )
        ])

    return qa_prompt

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
# 整理 source/tag
# *****
def source_file_organize(response):

    question_words = ["python","编程","语言"]
    
    # 记录用List
    source_files = []
    file_scores = {}

    for doc in response["source_documents"]:
        source_name = doc.metadata["file_name"]
        page_content = doc.page_content
        chunk_score = 0

        for kw in question_words:
            if kw in page_content:
                chunk_score += 1 
        if chunk_score > 0 :
            # 如果source_name不在sources_file这个dict里面
            if source_name not in file_scores :
                # 给sources_file[source_name]这个key新增一个value，并存放再这个dict里面
                file_scores[source_name] = chunk_score
            else :
                # 反之是sources_file[source_name]这个原来的value +1     
                # dirt[key]会自动取出values，这个是dict的固定写法
                file_scores[source_name] = file_scores[source_name] + chunk_score

    # 这里的写法和前面的enumerate（）很像，但那个是负责给数据，这里的items（）是负责给的key和value
    for source_name, count in file_scores.items():
        if count >= 1:           
            source_files.append(source_name)

    return source_files

# *****
# 定义判定chat模式提示词模板函数
# *****
def judge_prompt():
    prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                """
                "History" :{history}
                通过输入的问题，再结合上面的history，即聊天历史去，判断回复是否要参考上传文件
                只允许返回一个词，不要解释原因，不要添加其他内容：

                - rag：如果当前问题明显需要参考已上传文件，或者是在继续追问文件相关内容
                - normal：如果当前问题不需要参考已上传文件，或者与文件内容无明显关系

                只能返回：
                rag
                或
                normal

                """
            ),
            (
                "human", 
                """
                Question:{question}
                """
            )
        ])
    
    return prompt