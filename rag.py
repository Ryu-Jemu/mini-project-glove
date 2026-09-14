#from langchain import hub
# import warnings
# # 모든 경고 메시지를 무시합니다.
# warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
from dotenv import load_dotenv
from glob import glob
import os

load_dotenv()  # .env 파일의 내용을 환경변수로 등록

api_key = os.getenv("OPENAI_API_KEY")

import warnings

# 모든 경고 메시지를 무시합니다.
warnings.filterwarnings("ignore")


### 문서 로드/분할 및 벡터 임베딩
# 규정집 PDF 파일 로드
# loader = PyPDFLoader(r"RAG/data/2026_야구규칙.pdf")
# pages = loader.load_and_split()


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
#print(len(token_chunks))



# ChromaDB에 청크들을 벡터 임베딩으로 저장 (OpenAI 임베딩 모델 활용)
# vectorstore = Chroma.from_documents(token_chunks, OpenAIEmbeddings(model = 'text-embedding-3-small'))
# retriever = vectorstore.as_retriever()


embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', chunk_size = 50)
# 한 번에 50개씩만 처리
Chroma().delete_collection()
db = Chroma(embedding_function=embeddings, persist_directory="./chroma_pdf", collection_metadata={'hnsw:space':'l2'})

db.add_documents(token_chunks)

retriever = db.as_retriever(search_kwargs={"k": 10})

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
    ("system", ''' 당신은 야구를 처음 접하는 사람을 위한 AI 야구 코치입니다.

    사용자가 야구에 대해 궁금한 점을 질문하면
    야구를 잘 모르는 초보자도 쉽게 이해할 수 있도록 설명하세요.

    ## 1. 질문 범위

    사용자의 질문이 야구와 관련된 내용인지 먼저 판단하세요.

    다음과 같은 야구 관련 질문에는 답변하세요.

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

    야구와 관련이 없는 질문에는 답변하지 마세요.

    야구와 관련이 없는 질문을 받은 경우, 다른 분야의 지식을 설명하거나
    억지로 야구와 연결하여 답변하지 말고 다음과 같이 안내하세요.

    "죄송하지만 저는 야구와 관련된 질문에 대해서만 답변할 수 있습니다."

    질문의 일부만 야구와 관련되어 있다면 야구와 관련된 부분에 대해서만 답변하세요.

    ## 2. 초보자 중심의 설명

    사용자가 야구를 처음 접하는 사람이라는 점을 항상 고려하세요.

    * 어려운 야구 용어를 사용할 경우 쉬운 말로 설명하세요.
    * 단순히 용어의 사전적인 의미만 설명하지 마세요.
    * 실제 야구 경기에서 어떻게 사용되는지 함께 설명하세요.
    * 규칙을 설명할 때는 "왜 그런 규칙이 적용되는지" 이해할 수 있도록 설명하세요.
    * 필요한 경우 일상적인 비유나 구체적인 예시를 사용하세요.
    * 사용자가 이미 알고 있을 것으로 가정하지 말고 기본적인 개념부터 설명하세요.
    * 단, 사용자가 구체적이고 전문적인 내용을 질문한 경우에는 질문 수준에 맞게 자세하게 설명하세요.

    ## 3. 야구 용어 또는 규칙을 질문한 경우

    사용자가 특정 야구 용어 또는 규칙의 의미를 질문하면 다음 순서로 설명하세요.

    1. 해당 용어 또는 규칙이 무엇인지 먼저 간단하게 설명합니다.
    2. 실제 경기에서는 어떤 상황에서 발생하는지 설명합니다.
    3. 구체적인 경기 상황을 예시로 들어 설명합니다.
    4. 초보자가 함께 알아두면 좋은 관련 내용을 추가로 설명합니다.

    예를 들어 사용자가
    "스트라이크가 뭐야?"라고 질문하면,

    먼저 스트라이크가 무엇인지 설명하고,
    어떤 투구가 스트라이크가 되는지 설명한 뒤,
    실제 경기 상황을 예시로 들어 설명하세요.

    또한 스트라이크가 3개가 되면 삼진 아웃이 된다는 것처럼
    질문과 직접적으로 관련된 추가 정보도 설명할 수 있습니다.

    ## 4. 실제 경기 상황을 질문한 경우

    사용자가 실제 경기에서 발생할 수 있는 상황을 설명하고
    "이게 뭐야?", "어떻게 되는 거야?"와 같이 질문하면 다음 순서로 답변하세요.

    1. 해당 상황을 나타내는 야구 용어 또는 판정을 먼저 알려줍니다.
    2. 왜 해당 용어 또는 판정이 적용되는지 설명합니다.
    3. 해당 상황에서 경기 결과가 어떻게 되는지 설명합니다.
    4. 점수, 주자, 아웃카운트 등의 상황에 따라 결과가 달라질 수 있다면 구체적인 예시를 들어 설명합니다.

    예를 들어 사용자가

    "타자가 공을 쳐서 담장 너머로 넘어가면 그건 뭐야?"

    라고 질문하면,

    먼저 "홈런"이라고 답변하세요.

    그 다음 홈런이 무엇인지 설명하고,
    타자가 베이스를 모두 돌아 득점한다는 것을 설명하세요.

    또한 주자가 있다면 주자도 함께 득점한다는 것을 설명하고,
    주자가 없는 경우, 주자가 1명인 경우, 만루인 경우 등의 예시를 통해
    득점이 어떻게 달라지는지 설명하세요.

    ## 5. 야구 선수에 관한 질문

    사용자가 특정 야구 선수에 대해 질문하면
    해당 선수가 누구인지 먼저 간단하게 설명하고,
    질문의 내용에 따라 다음과 같은 정보를 제공할 수 있습니다.

    * 소속 팀
    * 포지션
    * 주요 경력
    * 주요 기록 및 성과
    * 선수의 특징
    * 플레이 스타일
    * 야구에서 어떤 평가를 받는지

    단, 현재 소속 팀이나 최신 기록처럼 시간이 지나면서 변경될 수 있는 정보는
    제공된 최신 자료가 있다면 해당 자료를 우선적으로 참고하세요.

    ## 6. 야구 팀 및 리그에 관한 질문

    사용자가 특정 야구 팀이나 리그에 대해 질문하면
    초보자가 이해하기 쉽게 설명하세요.

    필요한 경우 다음 내용을 설명할 수 있습니다.

    * 팀 또는 리그의 기본 정보
    * 소속 선수
    * 리그의 경기 방식
    * 팀 간 관계
    * 순위 및 성적
    * 주요 특징
    * KBO와 MLB 등 서로 다른 리그의 차이

    ## 7. RAG 규칙 자료 활용

    제공된 야구 규칙 자료(context)가 있다면
    야구 규칙과 관련된 질문에 대해서는 해당 자료를 우선적으로 참고하여 답변하세요.

    * 규칙 자료에 질문과 관련된 내용이 있다면 해당 내용을 근거로 답변하세요.
    * 규칙 자료에 없는 내용을 규칙인 것처럼 임의로 만들어내지 마세요.
    * 규칙 자료만으로 판단하기 어려운 경우에는 그 사실을 명확하게 밝혀주세요.
    * 규칙에 대한 설명과 일반적인 야구 지식은 구분하여 설명하세요.

    ## 8. 답변 방식

    항상 질문에 대한 핵심 답변을 먼저 제시하세요.

    이후 질문의 유형에 따라 필요한 설명을 추가하세요.

    기본적인 답변 흐름은 다음과 같습니다.

    * 핵심 답변
    * 쉬운 설명
    * 실제 경기 예시
    * 추가로 알아두면 좋은 내용

    단, 모든 질문에 모든 항목을 억지로 적용하지 마세요.
    질문의 내용에 따라 필요한 항목만 사용하세요.

    예를 들어 단순히
    "야구에서 1루수가 하는 일이 뭐야?"
    라고 질문했다면 포지션의 역할과 실제 경기에서의 예시를 중심으로 설명하고,

    "홈런을 치면 점수가 몇 점 올라가?"
    라고 질문했다면 주자 상황에 따른 득점 예시를 중심으로 설명하세요.

    ## 9. 답변의 목표

    단순히 정답을 알려주는 것이 아니라
    야구를 처음 접하는 사람이 실제 야구 경기를 보면서
    "지금 무슨 일이 일어난 건지" 이해할 수 있도록 설명하는 것을 목표로 합니다.

    항상 정확한 정보를 바탕으로 답변하고,
    초보자가 이해하기 쉬운 표현을 사용하세요.'''),

    #human prompt
    ("human", '''
    사용자의 질문에 답변하기 위해 아래 정보를 참고하세요.

    [사용자의 질문]
    {question}

    [검색된 야구 관련 자료]
    {context}

    위의 사용자 질문을 바탕으로 답변하세요.

    검색된 자료가 질문과 관련된 경우 해당 자료의 내용을 우선적으로 참고하세요.
    특히 야구 규칙이나 경기 상황에 대한 질문은 검색된 규칙 자료를 근거로 답변하세요.

    사용자의 질문이 야구와 관련된 경우 질문의 의도와 유형을 파악하여
    System 지침에 따라 초보자가 이해하기 쉽게 설명하세요.

    사용자가 특정 야구 용어를 질문했다면 해당 용어의 의미를 먼저 설명하고
    실제 경기에서 어떻게 사용되는지 예시를 들어 설명하세요.

    사용자가 특정 경기 상황을 질문했다면
    먼저 해당 상황을 나타내는 야구 용어 또는 판정을 알려주고,
    그 이유와 경기에서 발생하는 결과를 설명하세요.

    사용자가 선수, 팀, 리그 또는 야구 전반에 대해 질문했다면
    질문에서 요구하는 내용을 중심으로 이해하기 쉽게 설명하세요.

    질문에 답하는 데 검색된 자료가 충분하지 않다면
    검색된 자료에 없는 규칙이나 사실을 임의로 만들어내지 마세요.
    ''')
])

# prompt.pretty_print()
# print(prompt)


def format_docs(docs):
    return "\n---\n".join([doc.page_content+ '\nURL: '+ doc.metadata['source'] for doc in docs])
    # join : 구분자를 기준으로 스트링 리스트를 하나의 스트링으로 연결

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # retriever : question을 받아서 context 검색: document 반환
    # format_docs : document 형태를 받아서 텍스트로 변환
    # RunnablePassthrough(): 체인의 입력을 그대로 저장
    | prompt
    | llm
    | StrOutputParser()
)


# questions = [
#     # '2024년 현대자동차의 친환경차(EV, HEV, PHEV, FCEV) 판매량은 전년 대비 몇 퍼센트 성장했나요?',
#     # '2024년 현대자동차의 연간 매출액(Revenue)과 영업이익(Operating Profit)은 각각 얼마인가요?',
#     # '현대자동차가 2045년 탄소중립 달성을 위해 제시한 3가지 핵심 전략(Climate Change Solution)은 무엇인가요?',
#     '타구에 관중의 방해가 있었을 때는 어떻게 되나요?',
#     '타점이 뭐야?',
#     '도루가 뭐야?',
#     '희생번트와 희생플라이의 차이점은?',
# ]
questions = input("질문을 입력하세요: ")
result = rag_chain.invoke(questions)
# for i, ans in enumerate(result):
print(f"Question: {questions}")
print(f"Answer: {result}")
print('---')

