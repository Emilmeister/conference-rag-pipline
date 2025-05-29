import json

import streamlit as st
import asyncio
from agents import Agent, RunContextWrapper
from agents import set_default_openai_client, set_default_openai_api, set_trace_processors, Runner, trace, gen_trace_id, \
    function_tool
from agents.agent import StopAtTools
from openai.types.responses import EasyInputMessageParam
from phoenix.otel import register
from dotenv import load_dotenv
from agents.models import openai_provider
from openai import AsyncOpenAI
import logging
import os
import ssl
import requests
import numpy as np
import re

from typing_extensions import Any

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

ssl._create_default_https_context = ssl._create_unverified_context

PHOENIX_TRACE_URL = os.getenv("PHOENIX_TRACE_URL")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
RETRIEVER_URL = os.getenv("RETRIEVER_URL")

# configure the Phoenix tracer
set_trace_processors([])
tracer_provider = register(
    project_name=PHOENIX_PROJECT_NAME,
    endpoint=PHOENIX_TRACE_URL,
    auto_instrument=True
)

set_default_openai_client(AsyncOpenAI(base_url=OPENAI_API_URL, api_key=OPENAI_API_KEY, timeout=60 * 5))
set_default_openai_api('chat_completions')
openai_provider.DEFAULT_MODEL = DEFAULT_MODEL


def remove_think_tags(text: str) -> str:
    """Удаляет теги <think> и их содержимое из текста.

    Args:
        text: Исходный текст, который может содержать теги <think>.

    Returns:
        Текст без тегов <think> и их содержимого.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


# Using a separate event loop to run async code in Streamlit
class AsyncRunner:
    @staticmethod
    def run_async(func, *args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()


@function_tool
def rag_search(wrapper: RunContextWrapper[Any], query: str) -> str:
    """
    Используй для поиска релевантной информации в записях видеоконференций
    :param query: поисковый запрос
    :return: список релевантных документов

    Примеры query:
    1. Как зовут собаку Даши?
    2. К каким выводам пришли в результате обсуждения архитектуры нового автомобиля?
    3. Какие задачи стоят перед командой продаж?
    """
    return rag_search_inner(wrapper, query)


def rag_search_inner(wrapper: RunContextWrapper[Any], query: str) -> str:
    try:
        logging.info(f'rag_search_context: {json.dumps(wrapper.context)}')
        filters = {
            'query': query,
            'classic_search_num': 50,
            'vector_search_num': 50,
            'max_get_num': 7,
            'room_uuid': wrapper.context['room_uuid'] if 'room_uuid' in wrapper.context else ""
        }
        text_objs = requests.post(RETRIEVER_URL, json=filters).json()
        scores = [x['score'] for x in text_objs]
        texts_magic_sort = []
        idx_s = np.argsort(-np.array(scores))

        splitter = '%$$%#'

        available_room_uuids_strs = set([f"{x['room_uuid']}{splitter}{x['meeting_label']}" for x in text_objs])
        wrapper.context['available_room_uuids'] = []

        for available_room_uuids_str in available_room_uuids_strs:
            room_uuid = available_room_uuids_str.split(sep=splitter)[0]
            meeting_label = available_room_uuids_str.split(sep=splitter)[1]
            wrapper.context['available_room_uuids'].append({
                'room_uuid': room_uuid,
                'meeting_label': meeting_label
            })

        for idx in idx_s:
            doc_string = f"Часть записи видеоконференции c UUID {text_objs[idx]['room_uuid']}:\n{text_objs[idx]['parent_chunk']}"
            if idx % 2 == 1:
                texts_magic_sort = texts_magic_sort + [doc_string]
            else:
                texts_magic_sort = [doc_string] + texts_magic_sort

        context = '\n----\n\n'.join(texts_magic_sort)
        return "Полученные записи видеоконференций:\n\n" + context
    except Exception as e:
        logging.error(e)
        return "Произошла ошибка при запросе к сервису извлечения знаний из видеоконференций"


# Function to run a query with error handling
def run_agent_query(messages, context):
    try:
        async def run_query():
            agent = Agent(
                name="Videoconference helper",
                instructions="""
Ты помощник по поиску информации и ответам по видеоконференциям.
Используй tool rag_search для нахождения релевантной для ответа информации.
Если релевантной информации не нашлось, то сообщи об этом пользователю.
Отвечай точно на поставленный вопрос и не пиши лишнего что не относится к контексту вопроса.
                """,
                tools=[rag_search],
                tool_use_behavior=StopAtTools(stop_at_tool_names=['rag_search'] if 'room_uuid' not in context else [])
            )
            trace_id = gen_trace_id()
            with trace(workflow_name="Question Answering", trace_id=trace_id):
                result = await Runner.run(starting_agent=agent, input=messages, context=context)
                return result, trace_id

        return AsyncRunner.run_async(run_query)
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return f"Failed to process query: {str(e)}", None


def main():
    st.title("Помощник по содержанию видеоконференций")
    st.write("Задайте любой вопрос по содержанию прошедших видеоконференций")

    # Инициализация session_state для хранения истории
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'context' not in st.session_state:
        st.session_state.context = {}

    # Отображение истории переписки
    if st.session_state.history:
        st.subheader("История переписки")
        modified_history = st.session_state.history.copy()
        if 'selected_room_uuid_history_idx' in st.session_state.context:
            modified_history.insert(st.session_state.context['selected_room_uuid_history_idx'],
                                    {'role': 'selected_room_uuid'})
        for msg in modified_history:
            if 'role' in msg:
                if msg['role'] == 'user':
                    st.markdown(f"**Пользователь:**\n{msg['content']}")
                if msg['role'] == 'assistant':
                    if isinstance(msg['content'], str):
                        # st.markdown(f"**Помощник:**\n{msg['content']}")
                        st.markdown(f"**Помощник:**\n{remove_think_tags(msg['content'])}")
                        st.markdown("---")
                    else:
                        st.markdown(f"**Помощник:**\n{remove_think_tags(msg['content'][0]['text'])}")
                        st.markdown("---")
                if msg['role'] == 'selected_room_uuid':
                    st.markdown(f"### Выбранная конференция: [{st.session_state.context['meeting_label']}](https://live-lesson.ru/m/{st.session_state.context['room_uuid']})")
                    st.markdown("---")

    if 'available_room_uuids' in st.session_state.context and len(st.session_state.context['available_room_uuids']) > 1:
        room = st.selectbox(label="Выберите видеоконференцию: ",
                            index=None,
                            placeholder="...",
                            options=st.session_state.context['available_room_uuids'],
                            format_func=lambda x: f"{x['meeting_label']} ({x['room_uuid']})")
        if room:
            with st.spinner("Думаю..."):
                st.session_state.context['available_room_uuids'] = []
                st.session_state.context['room_uuid'] = room['room_uuid']
                st.session_state.context['meeting_label'] = room['meeting_label']
                st.session_state.context['selected_room_uuid_history_idx'] = len(st.session_state.history)

                st.session_state.history[-1]['output'] = rag_search_inner(RunContextWrapper(context=st.session_state.context), json.loads(st.session_state.history[-2]['arguments'])['query'])

                result, trace_id = run_agent_query(st.session_state.history, st.session_state.context)
                logging.info(f"result {result.to_input_list()}")
                st.session_state.history = result.to_input_list()

                if trace_id:
                    # Обновляем страницу для отображения обновлённой истории
                    st.rerun()
    else:
        # Input area for user queries
        query = st.text_area("Введите сообщение:", height=100, key="query_input")

        if st.button("Отправить"):
            if query:
                with st.spinner("Думаю..."):
                    st.session_state.history.append(EasyInputMessageParam(role='user', content=query))
                    result, trace_id = run_agent_query(st.session_state.history, st.session_state.context)
                    st.session_state.history = result.to_input_list()

                    if trace_id:
                        # Обновляем страницу для отображения обновлённой истории
                        st.rerun()


if __name__ == "__main__":
    main()
