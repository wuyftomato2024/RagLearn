from langchain_core.prompts import ChatPromptTemplate

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
# 定义判定chat模式提示词模板函数
# *****
def judge_prompt():
    prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                """
                "History" :{history}
                请判断当前问题应该属于哪一类。

                判断优先级：
                1. 先优先判断“当前问题本身”是不是明显在问上传文件、文档、资料、PDF、文件内容。
                如果是，优先返回 rag。
                2. 只有当当前问题本身不明显像文件问题时，才参考 history 判断是否属于 history。
                3. 如果当前问题既不像文件问题，也不像明显承接上一轮对话，就返回 normal。

                分类规则：
                - rag：
                当前问题明显是在问上传文件、文档、资料、PDF、文件内容，或者要求总结/概括上传文件内容。
                只要当前问题明显和文件内容有关，就优先返回 rag，不要因为 history 抢走判断。

                - history：
                当前问题本身不明显像文件问题，但明显是在继续上一轮对话，例如“上一个问题是什么”“你刚刚说的是什么意思”“继续刚才那个话题”。

                - normal：
                当前问题既不明显需要参考上传文件，也不明显需要参考上一轮对话。

                只允许返回一个词：
                rag
                或
                history
                或
                normal

                不要解释原因，不要添加其他内容。


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

# *****
# 命中文件来源本体关键词及模板
# *****
def chunk_hit_prompt():
    prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                """
                "History" :{history}
                "ai_text" :{ai_text}
                通过输入的问题，再结合上面的history，即聊天历史去，
                再结合ai_text这个文件块
                去判断这个问题应该参考ai_text里面的哪一个文件

                只允许返回一个文件名，不要解释原因，不要添加其他内容
                如果无法明确判断，就返回 None

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

# *****
# 判断「概括/总结整份上传文件的整体内容」模板
# *****
def summary_prompt():
    prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                """
                    请判断用户当前的问题，是否是在要求“概括/总结整份上传文件的整体内容”。

                    规则：
                    - 只有当用户明确想要“总结整体内容、概括主要内容、说明这份文件大意”时，才返回 True
                    - 如果用户是在问某个具体问题、某个具体细节、某一句内容、某个知识点，则返回 False
                    - 如果用户是在追问上一轮对话内容，也返回 False
                    - 如果只是和文件有关，但不是要求“整体概括”，也返回 False
                    - 不明确时，返回 False

                    下面这类问题返回 True：
                    - 总结一下上传文件
                    - 概括这份资料的内容
                    - 这份文件主要讲了什么
                    - 请简要总结这份文档

                    下面这类问题返回 False：
                    - 上一个问题问了什么
                    - 文件里有没有提到XX
                    - XX是什么意思
                    - 为什么这样写
                    - 这段内容说了什么
                    - 根据文件回答XX

                    只允许返回：
                    True
                    或
                    False

                    不要解释，不要添加其他内容。
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

# *****
# 根据「结果生成回答」模板
# *****
def summary_answer_prompt():
    prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                """
                根据输入的Result，做一个简洁概括

                """
            ),
            (
                "human", 
                """
                Result:{result}
                """
            )
        ])
    
    return prompt

def simple_normalChat():
    simple_system_message = """Please answer concisely and clearly.
                                Only answer the main point.
                                Do not give too many examples.
                                If the answer is not clearly stated in the provided context, do not make up information.
                                Instead, say that the context does not clearly mention it."""    
    return simple_system_message

def defult_normalChat():
    normal_system_message = """ 清楚回答问题
                            可适当说明
                            没有资料就明确说没有
                            不要编造"""
    return normal_system_message