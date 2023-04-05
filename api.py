from pathlib import Path
import logging
import time

import streamlit as st
import os
from qa import ChatPDF


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
DATA_DIR = Path("data/")
from dotenv import dotenv_values

config = dotenv_values(".env")
print(config["OPEN_API_KEY"])


def process_uploaded_document() -> None:
    LOGGER.info(f"Start processing the document.")
    if st.session_state.uploaded_file:
        with st.spinner("The document is being processed..."):
            timestamp = time.time()
            with open(
                os.path.join(DATA_DIR, st.session_state.uploaded_file.name), "wb"
            ) as f:
                f.write(st.session_state.uploaded_file.getbuffer())
            st.session_state.daia.process_document(
                file=os.path.join(DATA_DIR, st.session_state.uploaded_file.name)
            )
            LOGGER.info(f"It took {time.time() - timestamp} to process the document")


def answer_question() -> str:
    LOGGER.info(f"Start answer module.")
    if st.session_state.question and st.session_state.daia.docsearch:
        with st.spinner(
            "A moment, I'm looking at the document... It will take a few seconds."
        ):
            LOGGER.info(f"The question is: {st.session_state.question}")
            st.session_state.answer = st.session_state.daia.answer(
                question=st.session_state.question
            )
            # st.session_state.pages = st.session_state.daia.get_pages_from_sources(
            #     file=uploaded_file.getvalue()
            # )
    else:
        st.session_state.answer = "You have to give me a file to work with."


if "started" not in st.session_state:
    LOGGER.info("Starting the session.")
    st.session_state.daia = ChatPDF(config["OPEN_API_KEY"])
    st.session_state.started = True

st.title("Document AI Assistant")

st.sidebar.markdown("Hello, I'm ChatPDF, your document AI assistant.")
st.sidebar.markdown(
    "You can upload a PDF document and converse with me to obtain any answer. I love detailed questions. If you're not satisfied with my answers, complete your question with more details. **Have fun!**"
)
st.sidebar.markdown("**:yellow[IMPORTANT NOTE:]** *Don't put any sensitive documents.*")

uploaded_file = st.sidebar.file_uploader(
    "Input your file:", key="uploaded_file", on_change=process_uploaded_document
)

st.text_input(
    "What do you want to know about the document?",
    key="question",
    on_change=answer_question,
)

if "answer" in st.session_state:
    st.write(st.session_state.answer)
