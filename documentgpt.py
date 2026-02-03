from typing import LiteralString
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.storage import LocalFileStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.documents.base import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st
import hashlib, time, os
import openai

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📜",
)

if os.path.exists("./.cache") is False:
    os.makedirs("./.cache")
if os.path.exists("./.cache/files") is False:
    os.makedirs("./.cache/files")
if os.path.exists("./.cache/embeddings") is False:
    os.makedirs("./.cache/embeddings")


# OpenAI API key의 유효성을 검사합니다.
# 반환 값은 파일 업로더의 비활성화 여부를 결정합니다.
# disabled가 False일 때 버튼이 활성화되므로, API key가 유효하다면 False를 반환합니다.
def validate_key(api_key: str) -> bool:
    try:
        openai.OpenAI(api_key=api_key).models.list()
        return False
    except Exception as e:
        return True


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file: UploadedFile) -> VectorStoreRetriever:
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        add_start_index=True,
        chunk_size=600,
        chunk_overlap=100,
    )
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        api_key=API_KEY,
        model="text-embedding-3-large",
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
        key_encoder=lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def stream_data(message: str):
    for word in message.split(" "):
        yield word + " "
        time.sleep(0.01)


def send_message(message: str, role: str, save: bool = True) -> None:
    with st.chat_message(role):
        st.write_stream(stream_data(message))
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["message"])


def format_docs(docs: list[Document]) -> LiteralString:
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)


with st.sidebar:
    API_KEY = st.text_input("Enter your OpenAI API key", type="password")
    is_invalid = validate_key(API_KEY)
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file.",
        type=["pdf", "txt", "docx"],
        disabled=is_invalid,
    )
    st.link_button(
        "Github repo",
        "https://github.com/bioroid17/nomad-gpt-challenge",
    )
    with st.expander("View source code"):
        st.write(
            """
```python
from typing import LiteralString
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.storage import LocalFileStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.documents.base import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st
import hashlib, time
import openai


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📜",
)

if os.path.exists("./.cache") is False:
    os.makedirs("./.cache")
if os.path.exists("./.cache/files") is False:
    os.makedirs("./.cache/files")
if os.path.exists("./.cache/embeddings") is False:
    os.makedirs("./.cache/embeddings")


# OpenAI API key의 유효성을 검사합니다.
# 반환 값은 파일 업로더의 비활성화 여부를 결정합니다.
# disabled가 False일 때 버튼이 활성화되므로, API key가 유효하다면 False를 반환합니다.
def validate_key(api_key: str) -> bool:
    try:
        openai.OpenAI(api_key=api_key).models.list()
        return False
    except Exception as e:
        return True


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file: UploadedFile) -> VectorStoreRetriever:
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        add_start_index=True,
        chunk_size=600,
        chunk_overlap=100,
    )
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        api_key=API_KEY,
        model="text-embedding-3-large",
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
        key_encoder=lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def stream_data(message: str):
    for word in message.split(" "):
        yield word + " "
        time.sleep(0.01)


def send_message(message: str, role: str, save: bool = True) -> None:
    with st.chat_message(role):
        st.write_stream(stream_data(message))
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["message"])


def format_docs(docs: list[Document]) -> LiteralString:
    return "\\n\\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            ''',
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    '''
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
'''
)


with st.sidebar:
    API_KEY = st.text_input("Enter your OpenAI API key", type="password")
    is_invalid = validate_key(API_KEY)
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file.",
        type=["pdf", "txt", "docx"],
        disabled=is_invalid,
    )
    st.link_button(
        "Github repo",
        "https://github.com/bioroid17/nomad-gpt-challenge",
    )
    with st.expander("View source code"):
        st.write("Here comes the source code of this app:")

if file:
    llm = ChatOpenAI(
        temperature=0.1,
        api_key=API_KEY,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")
else:
    st.session_state["messages"] = []
```
"""
        )

if file:
    llm = ChatOpenAI(
        temperature=0.1,
        api_key=API_KEY,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")
else:
    st.session_state["messages"] = []
