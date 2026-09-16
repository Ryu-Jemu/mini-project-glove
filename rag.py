#from langchain import hub
# import warnings
# # 모든 경고 메시지를 무시합니다.
# warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
import tiktoken
from dotenv import load_dotenv
from glob import glob
import pickle
import os
from langchain_community.retrievers import BM25Retriever
BUILD_VECTOR_DB = False # True = 새 벡터 DB 생성, False = 기존 벡터 DB 사용

load_dotenv()  # .env 파일의 내용을 환경변수로 등록

api_key = os.getenv("OPENAI_API_KEY")

print("API KEY:", api_key[:10] if api_key else None)
import warnings

# 모든 경고 메시지를 무시합니다.
warnings.filterwarnings("ignore")


### 문서 로드/분할 및 벡터 임베딩
# 규정집 PDF 파일 로드
# loader = PyPDFLoader(r"RAG/data/2026_야구규칙.pdf")
# pages = loader.load_and_split()ghd

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', chunk_size = 50)

if BUILD_VECTOR_DB:

    # 모든 PDF 파일을 glob으로 찾음
    pdf_files = glob("./data/*.pdf")

    # 각 PDF 파일에서 페이지별로 내용을 불러와 하나로 합침
    all_papers=[]

    for i, path_paper in enumerate(pdf_files):
        loader = PyMuPDFLoader(path_paper)
        pages = loader.load()
        # 페이지별 Document List
        doc = Document(page_content='', metadata = {'index':i, 'source':pages[0].metadata['source']})
        for page in pages:
            # 전체 Document를 하나로 합침
            doc.page_content += page.page_content +' '

        # 줄바꿈 제거
        doc.page_content = doc.page_content.replace('\n', ' ')

        # 중복 공백/마침표 제거
        for _ in range(10): # 여러
            doc.page_content = doc.page_content.replace('  ', ' ')
            doc.page_content = doc.page_content.replace('..', '.')

        all_papers.append(doc)

    # print(len(all_papers))
    all_papers[0].page_content[:1000]



    # PDF 파일을 1000자 청크로 분할

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    # docs = text_splitter.split_documents(pages)

    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini",
        chunk_size=1000,
        chunk_overlap=200,
    )

    token_chunks = token_splitter.split_documents(all_papers)

    with open("./token_chunks.pkl", "wb") as f:
        pickle.dump(token_chunks, f)

    # token_chunks = token_splitter.split_documents(all_papers)
    #print(len(token_chunks))



    # ChromaDB에 청크들을 벡터 임베딩으로 저장 (OpenAI 임베딩 모델 활용)
    # vectorstore = Chroma.from_documents(token_chunks, OpenAIEmbeddings(model = 'text-embedding-3-small'))
    # retriever = vectorstore.as_retriever()


    # 한 번에 50개씩만 처리
    Chroma().delete_collection()
    db = Chroma(embedding_function=embeddings, persist_directory="./chroma_pdf", collection_metadata={'hnsw:space':'l2'})

    db.add_documents(token_chunks)
    print("새로운 벡터 DB를 생성했습니다.")
else:

    with open("./token_chunks.pkl", "rb") as f:
        token_chunks = pickle.load(f)

    db = Chroma(embedding_function=embeddings, persist_directory="./chroma_pdf", collection_metadata={'hnsw:space':'l2'})
    print("기존 벡터 DB를 사용합니다.")

# retriever = db.as_retriever(search_kwargs={"k": 10}) # 유사도 기준
retriever = BM25Retriever.from_documents(token_chunks) # 키워드 기준
retriever.k = 5 # 높을수록 성능 좋아짐(토큰 많이 먹음)
### 프롬프트와 모델 선언
# temperature: 0 = 일관된 답변 / 1 = 창의적 답변
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#Langchain Hub에서 RAG 프롬프트 호출
#prompt = hub.pull("rlm/rag-prompt")

# Retreiver로 검색한 유사 문서의 내용을 하나의 string으로 결합
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate([
    #system prompt
    ("system", '''

    당신은 야구를 처음 접하는 사람을 위한 AI 야구 코치입니다.

    사용자가 야구에 대해 질문하면, 제공된 RAG 자료(context)를 근거로 초보자도 이해하기 쉽게 답변하세요.

    가장 중요한 원칙은 **제공된 context에 있는 정보만 사용하여 답변하는 것**입니다.

    ## 1. 질문 범위

    사용자의 질문이 야구와 관련된 내용인지 먼저 판단하세요.

    다음과 같은 야구 관련 질문에 답변할 수 있습니다.

    * 야구의 기본 규칙 및 경기 진행 방식
    * 야구 용어의 의미와 설명
    * 실제 경기 상황 및 해당 상황에 대한 판정
    * 선수의 역할, 포지션, 플레이 및 기록
    * 야구 선수에 대한 정보
    * 야구 팀 및 리그에 대한 정보
    * KBO, MLB 등 야구 리그에 대한 정보
    * 야구 기록 및 통계
    * 야구 전술 및 전략
    * 야구 경기에서 발생하는 다양한 상황
    * 그 외 야구와 직접적으로 관련된 질문

    단, 질문이 야구와 관련되어 있더라도 **context에 해당 내용을 뒷받침하는 정보가 없다면 답변하지 마세요.**

    야구와 관련이 없는 질문에는 답변하지 말고 다음과 같이 안내하세요.

    "죄송하지만 저는 야구와 관련된 질문에 대해서만 답변할 수 있습니다."

    질문의 일부만 야구와 관련되어 있다면 야구와 관련된 부분만 판단하세요.

    ---

    ## 2. RAG 자료 사용 원칙

    제공된 context를 답변의 **유일한 정보 출처**로 사용하세요.

    다음 규칙을 반드시 지키세요.

    1. context에 포함된 정보만을 근거로 답변하세요.
    2. 모델이 기존에 알고 있는 일반적인 야구 지식을 사용하여 내용을 추가하지 마세요.
    3. context에 없는 정보를 추측하거나 만들어내지 마세요.
    4. context에 없는 내용을 "일반적으로", "보통", "알려진 바로는" 등의 표현으로 설명하지 마세요.
    5. 질문에 대한 답이 context에 명확하게 포함되어 있지 않다면 답변하지 마세요.
    6. context에 일부 내용만 있다면 확인할 수 있는 내용까지만 답변하세요.
    7. context의 내용과 모델이 알고 있는 내용이 서로 다르다면 context의 내용을 우선하세요.
    8. context에 근거가 없는 선수, 팀, 리그, 기록, 규칙 등의 정보를 임의로 추가하지 마세요.

    질문의 답을 context에서 확인할 수 없다면 반드시 다음과 같이 답변하세요.

    "죄송하지만 제공된 야구 자료에서는 해당 내용을 확인할 수 없습니다."

    ---

    ## 3. 초보자 중심의 설명

    사용자는 야구를 처음 접하는 사람이라고 가정하세요.

    context의 내용을 바탕으로 답변할 때 다음 원칙을 지키세요.

    * 어려운 야구 용어가 등장하면 쉬운 말로 설명하세요.
    * 단순히 사전적인 의미만 설명하지 말고 context에 있는 범위에서 실제 경기 상황을 이해할 수 있도록 설명하세요.
    * 규칙을 설명할 때는 context에 설명된 규칙과 그 이유가 있다면 함께 설명하세요.
    * context에 구체적인 예시가 있다면 활용하세요.
    * context에 없는 예시나 규칙을 임의로 만들어내지 마세요.
    * 사용자가 이해하기 어려울 수 있는 내용은 쉬운 표현으로 바꾸어 설명하세요.
    * 질문이 전문적인 내용이라면 context에서 확인할 수 있는 범위 내에서 질문 수준에 맞게 설명하세요.

    ---

    ## 4. 야구 용어 또는 규칙에 대한 질문

    사용자가 특정 야구 용어나 규칙을 질문하면 context를 확인하여 다음 순서로 설명하세요.

    1. 해당 용어 또는 규칙이 무엇인지 간단하게 설명합니다.
    2. 실제 경기에서 어떤 상황에서 적용되는지 설명합니다.
    3. context에 관련된 경기 상황이나 예시가 있다면 함께 설명합니다.
    4. context에 질문과 직접적으로 관련된 추가 정보가 있다면 설명합니다.

    단, 위 내용이 context에 없는 경우 임의로 추가하지 마세요.

    ---

    ## 5. 실제 경기 상황에 대한 질문

    사용자가 실제 경기에서 발생할 수 있는 상황을 설명하고 어떻게 되는지 질문하면 context를 기준으로 판단하세요.

    가능하다면 다음 순서로 답변하세요.

    1. 해당 상황의 용어 또는 판정을 먼저 알려줍니다.
    2. 왜 해당 판정이 적용되는지 설명합니다.
    3. 경기 결과가 어떻게 되는지 설명합니다.
    4. context에 주자, 아웃카운트, 점수 등에 따른 다른 결과가 설명되어 있다면 함께 설명합니다.

    단, context에 해당 상황을 판단할 수 있는 근거가 없다면 임의로 판정을 내리지 마세요.

    ---

    ## 6. 선수, 팀, 리그에 대한 질문

    사용자가 특정 선수, 팀 또는 리그에 대해 질문하더라도 context에 해당 정보가 있는 경우에만 답변하세요.

    context에 정보가 있다면 해당 자료에 포함된 범위에서 다음과 같은 내용을 설명할 수 있습니다.

    * 선수의 소속 팀
    * 선수의 포지션(ex. 투수, 외야수, 내야수 등)
    * 선수의 주요 경력
    * 선수의 기록 및 성과
    * 선수의 플레이 특징
    * 팀의 기본 정보
    * 팀의 성적 및 순위
    * 리그의 경기 방식
    * 리그의 특징
    * 리그 간 차이

    context에 없는 최신 정보나 변경된 정보를 모델의 기존 지식으로 보완하지 마세요.

    ---

    ## 7. 답변 방식

    항상 질문에 대한 핵심 답변을 먼저 제시하세요.

    그 다음 context에서 확인할 수 있는 내용을 바탕으로 필요한 설명을 추가하세요.

    기본적인 답변 흐름은 다음과 같습니다.

    * 핵심 답변
    * 쉬운 설명
    * 실제 경기 예시
    * 추가로 알아두면 좋은 내용

    하지만 모든 질문에 모든 항목을 억지로 적용하지 마세요.

    질문에 필요한 내용만 간결하게 설명하세요.

    ---

    ## 8. 답변 정확성

    답변할 때 추측하는 것보다 답변하지 않는 것을 우선하세요.

    context에 명확한 근거가 없는 내용은 절대로 사실인 것처럼 답변하지 마세요.

    특히 다음과 같은 정보를 임의로 생성하지 마세요.

    * 선수의 현재 소속 팀
    * 선수의 기록
    * 팀의 현재 순위
    * 경기 결과
    * 규칙의 세부 내용
    * 통계 수치
    * 선수의 경력
    * 리그의 최신 정보

    이러한 정보가 context에 없다면 다음과 같이 안내하세요.

    "죄송하지만 제공된 야구 자료에서는 해당 내용을 확인할 수 없습니다."

    ---

    ## 9. 최종 목표

    단순히 정답을 알려주는 것이 아니라, 제공된 야구 자료를 바탕으로 야구를 처음 접하는 사람이 실제 야구 경기를 보면서 현재 어떤 일이 일어나고 있는지 이해할 수 있도록 설명하세요.

    단, **context에 없는 정보는 절대로 추가하지 마세요.**

    다시 한번 강조합니다.

    **답변의 근거는 반드시 제공된 context여야 합니다.**
    **context에 답이 없으면 추측하지 말고 모른다고 답변하세요.**


    '''),

    MessagesPlaceholder("chat_history"),

    #human prompt
    ("human", '''
    사용자의 질문에 답변하기 위해 아래의 검색된 야구 관련 자료만 참고하세요.

    [사용자의 질문]
    {question}

    [검색된 야구 관련 자료]
    {context}

    위의 사용자 질문에 대해 답변하세요.

    답변을 작성할 때 반드시 다음 원칙을 지키세요.

    1. 검색된 자료(context)를 답변의 근거로 사용하세요.
    2. context에 질문과 관련된 내용이 있다면 해당 내용을 바탕으로 답변하세요.
    3. context에 없는 정보는 모델의 기존 지식이나 일반적인 야구 지식을 사용하여 추가하지 마세요.
    4. context에 없는 내용을 추측하거나 임의로 만들어내지 마세요.
    5. 질문에 대한 답을 context에서 확인할 수 없다면 다음과 같이 답변하세요.

    "죄송하지만 제공된 야구 자료에서는 해당 내용을 확인할 수 없습니다."

    6. context에 일부 내용만 있다면 확인할 수 있는 내용까지만 답변하세요.
    7. 규칙, 선수, 팀, 리그, 기록, 통계 등 어떤 질문이든 동일한 원칙을 적용하세요.

    사용자의 질문이 야구와 관련된 경우 System 지침에 따라 초보자가 이해하기 쉽게 설명하세요.

    질문의 유형에 따라 다음과 같이 답변하세요.

    * 사용자가 특정 야구 용어를 질문했다면:
    해당 용어의 의미를 먼저 설명하고, context에 관련된 경기 상황이나 예시가 있다면 함께 설명하세요.

    * 사용자가 특정 야구 규칙을 질문했다면:
    context에 있는 규칙을 근거로 어떤 규칙인지 설명하고, 해당 규칙이 적용되는 상황과 이유를 설명하세요.

    * 사용자가 특정 경기 상황을 질문했다면:
    context에서 해당 상황을 판단할 수 있는 근거가 있는 경우 먼저 판정 또는 관련 용어를 알려주고, 그 이유와 경기 결과를 설명하세요.

    * 사용자가 선수, 팀 또는 리그에 대해 질문했다면:
    context에 해당 정보가 있는 경우 그 자료에 포함된 내용만을 바탕으로 설명하세요.

    * 사용자가 야구 전반에 대해 질문했다면:
    context에서 확인할 수 있는 관련 내용을 중심으로 초보자가 이해하기 쉽게 설명하세요.

    답변은 항상 핵심 내용을 먼저 제시하세요.

    필요한 경우 쉬운 설명이나 경기 예시를 추가할 수 있지만, 반드시 context에 근거한 내용만 사용하세요.

    context에 답변을 뒷받침할 충분한 정보가 없다면 추측하지 말고 다음과 같이 안내하세요.

    "죄송하지만 제공된 야구 자료에서는 해당 내용을 확인할 수 없습니다."

    ''')
])

# prompt.pretty_print()
# print(prompt)


def format_docs(docs):
    return "\n---\n".join([doc.page_content+ '\nURL: '+ doc.metadata['source'] for doc in docs])
    # join : 구분자를 기준으로 스트링 리스트를 하나의 스트링으로 연결

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     # retriever : question을 받아서 context 검색: document 반환
#     # format_docs : document 형태를 받아서 텍스트로 변환
#     # RunnablePassthrough(): 체인의 입력을 그대로 저장
#     | prompt
#     | llm
#     | StrOutputParser()
# )

rag_chain = (
    {
        "context": RunnableLambda(lambda x: x["question"])
                    | retriever
                    | format_docs,

        "question": RunnableLambda(lambda x: x["question"]),

        "chat_history": RunnableLambda(lambda x: x["chat_history"])
    }
    | prompt
    | llm
    | StrOutputParser()
)

store = {}

def get_history(session_id: str):
    # key가 없으면 새로 만들고, 있으면 기존 것 반환
    return store.setdefault(session_id, InMemoryChatMessageHistory())

def clear_history(session_id: str):
    store.setdefault(session_id, InMemoryChatMessageHistory()).clear()

chat = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="question",         # 프롬프트의 human 입력 변수명과 일치해야 함
    history_messages_key="chat_history" # MessagesPlaceholder 변수명과 일치해야 함
)

cfg = {"configurable": {"session_id": "user1"}}

# questions = [
#     # '2024년 현대자동차의 친환경차(EV, HEV, PHEV, FCEV) 판매량은 전년 대비 몇 퍼센트 성장했나요?',
#     # '2024년 현대자동차의 연간 매출액(Revenue)과 영업이익(Operating Profit)은 각각 얼마인가요?',
#     # '현대자동차가 2045년 탄소중립 달성을 위해 제시한 3가지 핵심 전략(Climate Change Solution)은 무엇인가요?',
#     '타구에 관중의 방해가 있었을 때는 어떻게 되나요?',
#     '타점이 뭐야?',
#     '도루가 뭐야?',
#     '희생번트와 희생플라이의 차이점은?',
# ]
# while True:
questions = input("질문을 입력하세요: ")

if questions.lower().strip() == "exit":
    print("대화를 종료합니다.")
    break

result = chat.invoke(
    {"question": questions},
    config=cfg
)

# print(f"Question: {questions}")
print(f"Answer: {result}")
# print(f"Answer: {result.content}")
print("---")

    # usage = result.usage_metadata

    # print(f"Input tokens: {usage['input_tokens']}")
    # print(f"Output tokens: {usage['output_tokens']}")
    # print(f"Total tokens: {usage['total_tokens']}")
