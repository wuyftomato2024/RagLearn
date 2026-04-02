import streamlit as st
import requests

# 标题
st.header("ai reader")

with st.sidebar:
    openai_api_key = st.text_input("ApiKey",type="password")

# 输入组件
question = st.text_input("Question")
top_k = 2
# accept_multiple_files=True是允许复数文件的意思
upload_file = st.file_uploader("UploadFile",type=["txt","pdf"],accept_multiple_files=True)

# 先创建一个变量，用来存储判断的结果，因为判断式，本身就会生成布尔值，可以存到变量里面，不一定是要写if的，if只是一个分支
disabled_flag = not question or not openai_api_key

if disabled_flag :
    st.info("please input question and openai_api_key")

# 如果disabled_flag等于true 那这里的disabled就等于true，就把按钮的开关给关上了
search_button = st.button("search" ,disabled= disabled_flag)

# 设定要传入的值
# "后端要的名字" : 前端当前拿到的值
data = {
    "question" : question ,
    "top_k" :top_k ,
    "openai_api_key" : openai_api_key
}

files = []
for file in upload_file :
    files.append(
        # "后端接收这个文件的名字", "这个文件本体"
        ("upload_file",(file.name,file.getvalue(), file.type))
        )

if search_button :
    with st.spinner("thinking..."):
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            data=data,
            files=files
        )

        result = response.json()

        if result["status"] == "ok":
            st.subheader("answer")
            st.write(result["data"]["answer"])
            tags = result["data"]["tag"]
            if not tags :
                st.info("tags was empty")
            else :
                st.subheader("tag")
                for tag in tags :
                    st.write("information come from :"+ tag)
        else :
            st.error(result["detail"])

