from typing import override
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
)
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from openai import AssistantEventHandler, conversations
import streamlit as st
import openai, json, time

# # First, we create a EventHandler class to define
# # how we want to handle the events in the response stream.

PROMPT_NAME = "Search Prompt"


st.set_page_config(
    page_title="Prompt",
    page_icon="💬",
)


# OpenAI API key의 유효성을 검사합니다.
# 반환 값은 파일 업로더의 비활성화 여부를 결정합니다.
# disabled가 False일 때 버튼이 활성화되므로, API key가 유효하다면 False를 반환합니다.
def validate_key(api_key: str) -> bool:
    try:
        openai.OpenAI(api_key=api_key).models.list()
        return False
    except Exception as e:
        return True


with st.sidebar:
    API_KEY = st.text_input("Enter your OpenAI API key", type="password")
    is_invalid = validate_key(API_KEY)
    st.link_button(
        "Github repo",
        "https://github.com/bioroid17/nomad-gpt-challenge",
    )
    with st.expander("View source code"):
        st.markdown(
            """
"""
        )


# Tools
def wikipedia_search(inputs):
    query = inputs["query"]
    wp = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    print(f"using wikipedia_search, query: {query}")
    return wp.run(query)


def duckduckgo_search(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
    urls = ddg.run(query)
    print(f"using duckduckgo_search, query: {query}")
    return urls


functions_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
}

functions = [
    {
        "type": "function",
        "name": "wikipedia_search",
        "description": "Given the query, return the search result from wikipedia.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The keyword of the information you want to search.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "duckduckgo_search",
        "description": "Given the query, return the search result from duckduckgo.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The keyword of the information you want to search.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def get_response(conversation_id, content):
    return openai.responses.create(
        conversation=conversation_id,
        model="gpt-4.1-2025-04-14",
        input=[{"role": "user", "content": content}],
        tools=functions,
        tool_choice="required",
    )


def get_response2(conversation_id, result, function_call):
    return openai.responses.create(
        conversation=conversation_id,
        model="gpt-4.1-2025-04-14",
        input=[
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": result,
            },
        ],
    )


def stream_data(message: str):
    for word in message.split(" "):
        yield word + " "
        time.sleep(0.01)


def insert_message(message: str, role: str, save: bool = True) -> None:
    with st.chat_message(role):
        st.write_stream(stream_data(message))
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history() -> None:
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.write(message["message"])


if not is_invalid:
    openai.api_key = API_KEY
    if "conversation" not in st.session_state:
        conversation = openai.conversations.create()
        st.session_state["conversation"] = conversation
    else:
        conversation = st.session_state["conversation"]

    paint_history()
    content = st.chat_input("What do you want to search?", disabled=is_invalid)
    if content:
        insert_message(content, "user")
        response = get_response(conversation.id, content)
        if response.output[0].type == "function_call":
            print("response type: function call")

            function_call = response.output[0]
            function_name = function_call.name
            arguments = json.loads(function_call.arguments)

            # Response API의 경우 function call 처리를 위해 function call 결과를 다시 OpenAI에 넣어줘야 함
            if function_name == "wikipedia_search":
                result = wikipedia_search(arguments)
                response2 = get_response2(conversation.id, result, function_call)
            elif function_name == "duckduckgo_search":
                result = duckduckgo_search(arguments)
                response2 = get_response2(conversation.id, result, function_call)
            else:
                response2 = get_response(
                    conversation.id, f"Wrong function call: {function_name}"
                )
            insert_message(response2.output_text, "ai")
        else:
            print("response type: message")
            insert_message(response.output_text, "ai")
else:
    st.session_state["messages"] = []
    st.sidebar.warning("Input OpenAI API Key.")
