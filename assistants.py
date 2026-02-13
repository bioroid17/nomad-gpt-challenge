from typing import override
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from openai import AssistantEventHandler
import streamlit as st
import openai, json


ASSISTANT_NAME = "Search Assistant"


class EventHandler(AssistantEventHandler):

    message = ""
    run_id = ""
    thread_id = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message)

    def on_event(self, event):
        if event.event == "thread.run.created":
            self.run_id = event.data.id
            self.thread_id = event.data.thread_id

        if event.event == "thread.run.requires_action":
            submit_tool_outputs(self.run_id, self.thread_id)


st.set_page_config(
    page_title="Assistant",
    page_icon="🧰",
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
```python
from typing import override
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from openai import AssistantEventHandler
import streamlit as st
import openai, json


ASSISTANT_NAME = "Search Assistant"


class EventHandler(AssistantEventHandler):

    message = ""
    run_id = ""
    thread_id = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message)

    def on_event(self, event):
        if event.event == "thread.run.created":
            self.run_id = event.data.id
            self.thread_id = event.data.thread_id

        if event.event == "thread.run.requires_action":
            submit_tool_outputs(self.run_id, self.thread_id)


st.set_page_config(
    page_title="Assistant",
    page_icon="🧰",
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
        st.markdown("Here comes the source code")


# Tools
def wikipedia_search(inputs):
    query = inputs["query"]
    wp = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wp.run(query)


def duckduckgo_search(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchRun()
    urls = ddg.run(query)
    return urls


functions_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Given the query, return the search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The keyword of the information you want to search.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Given the query, return the search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The keyword of the information you want to search.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


#### Utilities
def get_run(run_id, thread_id):
    return openai.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = openai.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with openai.beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def insert_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(thread_id):
    messages = get_messages(thread_id)
    for message in messages:
        insert_message(
            message.content[0].text.value,
            message.role,
        )


if "run" in st.session_state:
    pass

if not is_invalid:
    openai.api_key = API_KEY
    if "assistant" not in st.session_state:
        assistants = openai.beta.assistants.list(limit=10)
        for a in assistants:
            if a.name == ASSISTANT_NAME:
                assistant = openai.beta.assistants.retrieve(a.id)
                break
        else:
            assistant = openai.beta.assistants.create(
                name=ASSISTANT_NAME,
                instructions="You help users do research on the given query using search engines. You give users the summarization of the information you got.",
                model="gpt-4o-mini",
                tools=functions,
            )
        thread = openai.beta.threads.create()
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread
    else:
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]

    paint_history(thread.id)
    content = st.chat_input("What do you want to search?", disabled=is_invalid)
    if content:
        send_message(thread.id, content)
        insert_message(content, "user")

        with st.chat_message("assistant"):
            with openai.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()
else:
    st.sidebar.warning("Input OpenAI API Key.")

```
"""
        )


# Tools
def wikipedia_search(inputs):
    query = inputs["query"]
    wp = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wp.run(query)


def duckduckgo_search(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchRun()
    urls = ddg.run(query)
    return urls


functions_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Given the query, return the search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The keyword of the information you want to search.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Given the query, return the search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The keyword of the information you want to search.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


#### Utilities
def get_run(run_id, thread_id):
    return openai.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = openai.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with openai.beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def insert_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(thread_id):
    messages = get_messages(thread_id)
    for message in messages:
        insert_message(
            message.content[0].text.value,
            message.role,
        )


if "run" in st.session_state:
    pass

if not is_invalid:
    openai.api_key = API_KEY
    if "assistant" not in st.session_state:
        assistants = openai.beta.assistants.list(limit=10)
        for a in assistants:
            if a.name == ASSISTANT_NAME:
                assistant = openai.beta.assistants.retrieve(a.id)
                break
        else:
            assistant = openai.beta.assistants.create(
                name=ASSISTANT_NAME,
                instructions="You help users do research on the given query using search engines. You give users the summarization of the information you got.",
                model="gpt-4o-mini",
                tools=functions,
            )
        thread = openai.beta.threads.create()
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread
    else:
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]

    paint_history(thread.id)
    content = st.chat_input("What do you want to search?", disabled=is_invalid)
    if content:
        send_message(thread.id, content)
        insert_message(content, "user")

        with st.chat_message("assistant"):
            with openai.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()
else:
    st.sidebar.warning("Input OpenAI API Key.")
