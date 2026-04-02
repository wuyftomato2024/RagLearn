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

db = None
# 嵌入模型  把每个文本块转成向量（变成数字）
embedding_model = OpenAIEmbeddings()


# 因为fastapi用的是异步上传，所以这里要加上异步“async” 
async def ragChat(question , memory ,upload_file ,openai_api_key ,top_k):
    # 定义模型
    model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key = openai_api_key)
    # 判断top_k的值，如果非法就报错
    if top_k < 1 or top_k > 3 :
        raise HTTPException(status_code=400 , detail="top_k must be between 1 and 3")
    
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
                # 创建一个对话提示词模板     用""""内容""""的写法是因为，写多行
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
    
    

    # 调用全局函数的db而不是重新生成一个db
    global db

    # # 如果上传文件不为空,且上传文件名不等于空
    # if upload_file is not None and upload_file.filename != "" :
    #     # 上传文件   并且读取内容 变成二进制 并且存进去file_content。await是异步处理的用法，和上面的async是配套的
    #     file_content = await upload_file.read()
    #     # 防止上传文件没有内容的报错
    #     if not file_content :
    #         raise HTTPException(status_code=400 ,detail="please upload file")
        
    #     filename = upload_file.filename

    #     if filename.endswith(".txt"):    
    #         # 定义上传文件 生成的临时文件（这一步只是定义，没做任何的处理）  temp.pdf是生成在当前路径的文件名，可以自己改
    #         temp_file_path = "temp.txt"
    #         # 把刚才读到的上传文件，临时保存成一个本地 txt 文件
    #         with open(temp_file_path,"wb") as temp_file :
    #             temp_file.write(file_content)
    #         # 创建一个txt读取工具
    #         loader = TextLoader(temp_file_path ,encoding="utf-8")
    #     elif filename.endswith(".pdf"):
    #         # 定义上传文件 生成的临时文件（这一步只是定义，没做任何的处理）  temp.pdf是生成在当前路径的文件名，可以自己改
    #         temp_file_path = "temp.pdf"
    #         # 把刚才读到的上传文件，临时保存成一个本地 pdf 文件
    #         with open(temp_file_path,"wb") as temp_file :
    #             temp_file.write(file_content)
    #         loader = PyPDFLoader(temp_file_path)
    #     else :
    #         raise HTTPException(status_code=400 ,detail="only txt and pdf are supported ")

    #     # 真的把 txt 内容加载出来，变成 LangChain 能处理的 document 列表
    #     docs = loader.load()
    #     # 设置如何分割 txt文件     
    #     text_splitters = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,
    #         chunk_overlap=50,
    #         separators=["\n", "。", "!", "？", ",", "、", ""]
    #     )
    #     # 按照刚才的规则，把 docs 切成很多文档块。 调用了text_splitters里面的split_documents（）方法，切割docs
    #     texts = text_splitters.split_documents(docs)    
    #     # 文档向量化 ，存入数据库 放入(分割好的文件块和嵌入模型，把texts给变成数字)
    #     db = FAISS.from_documents(texts ,embedding_model)
    # # 如果上传文件为空，且db为None ，就会报错
    # elif db is None :
    #     raise HTTPException(status_code=400 , detail="please upload a txt or pdf file first")

    # 复数写法
    if upload_file :
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
        db = FAISS.from_documents(docs_list,embedding_model)

    elif db is None :
        raise HTTPException(status_code=400 , detail="please upload a txt or pdf file first")
    
    # 把数据库变成一个“检索器”。后面的search_kwargs是固定写法，是搜索参数的意思，必须是要写成字典的形式
    db_retriever = db.as_retriever(search_kwargs= {"k": top_k}) 
    # 创建一个“带记忆 + 会检索2资料”的问答链。
    
    qa = ConversationalRetrievalChain.from_llm(
        llm = model ,
        retriever = db_retriever,
        memory = memory ,
        return_source_documents = True ,
        combine_docs_chain_kwargs = {"prompt":qa_prompt}
    )
    response = qa.invoke({"question" : question})

    # 先创建history列表的空值
    history_list = []
    # for循环
    for msg in memory.chat_memory.messages :
        # isinstance()为判断的用法，判断（a等于b）
        if isinstance(msg,HumanMessage):
            # 如果a等于b的话就往history_list里面添加以下字典
            history_list.append(HistoryItem(role = "human",content = msg.content))
        elif    isinstance(msg,AIMessage):
            history_list.append(HistoryItem(role = "ai" , content = msg.content))
    
    # 记数用的dict
    sources_file = {}
    # 记录用List
    source_file = []
    # 当前最大count

    for doc in response["source_documents"]:
        source_name = doc.metadata["file_name"]
        # 如果source_name不在sources_file这个dict里面
        if source_name not in sources_file :
            # 给sources_file[source_name]这个key新增一个value，并存放再这个dict里面
            sources_file[source_name] = 1
        else :
            # 反之是sources_file[source_name]这个原来的value +1     
            # dirt[key]会自动取出values，这个是dict的固定写法
            sources_file[source_name] = sources_file[source_name] +1

    # 这里的写法和前面的enumerate（）很像，但那个是负责给数据，这里的items（）是负责给的key和value
    for source_name, count in sources_file.items():
        if count >= 1:           
            source_file.append(source_name)

    # print(response["source_documents"])
    # print(sources_file)
    # print(response["answer"])
    # print(repr(response["answer"]))

    # return ChatResponse(
    #     answer = response["answer"] ,
    #     chatHistory = history_list ,
    #     tag = source_file
    # )
    
    return ApiResponse(
        status = "ok",
        data = ChatResponse(
            answer = response["answer"] ,
            chatHistory = history_list ,
            tag = source_file)
    )


# 回答的是LLM，LLM拿到了检索后的相关内容再来进行回答