import streamlit as st
from operator import itemgetter
import subprocess
import math
import os
from pydub import AudioSegment
import glob
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.schema.runnable.base import RunnableLambda
from langchain.prompts import MessagesPlaceholder
from langchain_core.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
)

st.title("미팅 분석 GPT")

st.markdown(
    """
안녕하세요. 당신의 미팅 비서GPT입니다. 저에게 영상을 주시면 미팅 내용을 기록 후 요약해드립니다.
미팅 내에 있었던 모든 일들에 대해 물어보세요.
"""
)
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"


has_transcript = os.path.exists("./.cache/podcast.txt")
has_summary = os.path.exists("./.cache/summary.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = ["/opt/homebrew/bin/ffmpeg", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)

def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk{i}.mp3", format="mp3")

@st.cache_resource()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever



@st.cache_resource()
def transcribe_chunks(chunk_folder, destination):
    files = glob.glob(f"{chunk_folder}/*.mp3") #폴더 내에 mp3 확장자를 가진 파일을 찾는 것
    final_transcript = "" #초기화
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])

def save_summary(file_name, summary):
    summary_path = f"./.cache/summary/{file_name}.txt"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as file:
        file.write(summary)

def load_summary(file_name):
    summary_path = f"./.cache/summary/{file_name}.txt"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as file:
            return file.read()
    return None

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
if video:
    video_path = f"./.cache/{video.name}"
    audio_path = video_path.replace("mp4", "mp3")
    chunks_folder = "./.cache/chunks"
    transcript_path = video_path.replace("mp4", "txt")
    with st.status("Loading video...") as status:
        with open(video_path, "wb") as file:
            file.write(video.read())
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)
    transcript_tab, summary_tab, qa_tab = st.tabs(
        ["Transcript", "Summary", "Q&A"],
    )


    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        file_name = os.path.basename(transcript_path).replace(".txt", "")
        saved_summary = load_summary(file_name)

        if saved_summary:
            st.write(saved_summary)
        else:
            start = st.button("요약 시작")

            if start:
                loader = TextLoader(transcript_path)
                docs = loader.load_and_split(text_splitter=splitter)

                first_summary_prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 문서 요약 전문가입니다.
                    다음 문서를 정확하게 요약하세요.
                    "{text}"
                    간결한 요약:
                """
                )
                first_summary_chain = first_summary_prompt | llm | StrOutputParser()
                summary = first_summary_chain.invoke(
                    {"text": docs[0].page_content},
                )

                refine_prompt = ChatPromptTemplate.from_template(
                    """
                    최종 요약을 작성하는 것이 당신의 임무입니다.
                    우리는 특정 지점까지의 기존 요약을 제공했습니다: {existing_summary}
                    추가적인 맥락을 통해 기존 요약을 개선할 기회가 있습니다.
                    ------------
                    {context}
                    ------------
                    새로운 맥락을 고려하여 기존 요약을 개선하세요.
                    맥락이 유용하지 않으면 기존 요약을 반환하세요.
                    """
                )
                refine_chain = refine_prompt | llm | StrOutputParser()

                with st.status("Summarizing...") as status:
                    for i, doc in enumerate(docs[1:]):
                        status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
                        summary = refine_chain.invoke(
                            {
                                "existing_summary": summary,
                                "context": doc.page_content,
                            }
                        )
                        st.write(summary)

                st.write(summary)
                save_summary(file_name, summary)

    with qa_tab:
        retriever = embed_file(transcript_path)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Answer the question using ONLY the following context.
                    If you don't know the answer just say you don't know. DON'T make anything up.

                    Context: {context}
                    """,
                ),
                ("human", "{question}"),
            ]
        )

        message = st.text_input("Ask anything about your video...")

        if message:
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | qa_prompt
                | llm
            )

            with st.chat_message("ai"):
                chain.invoke(message).content
