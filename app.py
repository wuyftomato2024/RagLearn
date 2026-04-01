import streamlit as st
import requests

st.header("ai reader")

question = st.text_input("please input your question")
top_k = st.number_input("please input your top_k")
# accept_multiple_files=True是允许复数文件的意思
upload_file = st.file_uploader("please input your upload_file",type=["txt","pdf"],accept_multiple_files=True)
openai_api_key = st.text_input("please input your api_key",type="password")

search_button = st.button("search")

if not question or openai_api_key:
    st.info("please input your secret key")

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
    response = requests.post(
        "http://127.0.0.1:8000/chat",
        data=data,
        files=files
    )

    result = response.json()

    if result["status"] == "ok":
        st.subheader("answer")
        st.write(result["data"]["answer"])
        st.subheader("tag")
        st.write(result["data"]["tag"])
    else :
        st.error(result["detail"])

