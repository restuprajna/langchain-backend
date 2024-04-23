import os
import sys

from langchain_community.retrievers import GoogleVertexAISearchRetriever
# from langchain_community.chat_models import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, StringPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, TypedDict, Dict, Any, List
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
    RetryOutputParser,
    ResponseSchema,
    StructuredOutputParser,
    RetryOutputParser
)
from langchain_mongodb import MongoDBAtlasVectorSearch


from dotenv import load_dotenv

# from langsmith import Client


load_dotenv

from pymongo import MongoClient

# define google api and vertex model
google_api: str = os.environ["GOOGLE_API_KEY"]
vertex_model: str = os.environ["vertex_model"]

# initialize MongoDB python client
atlas_cluster_uri: str = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
client = MongoClient(atlas_cluster_uri)

DB_NAME: str = os.environ["DB_NAME"]
EMBEDDING_COLLECTION_NAME: str = os.environ["EMBEDDING_COLLECTION"]
ATLAS_VECTOR_SEARCH_INDEX_NAME: str = os.environ["ATLAS_VECTOR_SEARCH_INDEX_NAME"]

MONGODB_COLLECTION = client[DB_NAME][EMBEDDING_COLLECTION_NAME]


# setup embedding model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding_model: str = os.environ["embedding_model"]
google_embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=google_api)


# declare vector store
vector_store = MongoDBAtlasVectorSearch(
    # embedding=OpenAIEmbeddings(disallowed_special=(), openai_api_key=openai_api_key),
    embedding=google_embeddings,
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

safety_settings_NONE=safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

google_api: str = os.environ["GOOGLE_API_KEY"]
vertex_model: str = os.environ["vertex_model"]
llm = ChatGoogleGenerativeAI(temperature=1.0,
                            model=vertex_model, 
                            google_api_key=google_api, 
                            safety_settings=safety_settings_NONE)

# setup retriever
k = 2
retriever = vector_store.as_retriever(
    search_kwargs={
        'post_filter_pipeline': [ { '$limit': k } ]
    }
)


StringPromptTemplate=(
    """
{{ 
    "name": "Bahasa Indonesia",
    "level_pendidikan" : "Kelas 12"
    "goals": "mengembangkan pemahaman mendalam siswa terhadap karya sastra klasik dan modern, meningkatkan kemampuan analisis teks, serta mengasah keterampilan menulis kreatif, berbicara, dan berdiskusi, sambil menanamkan nilai-nilai budaya dan moral, serta mengembangkan minat dan apresiasi mereka terhadap sastra Indonesia."
 }}
"""
)

response_format_prompt = PromptTemplate.from_template(StringPromptTemplate)



template = (
    """
Selalu ikuti instruksi berikut: kamu adalah asisten yang akan membantu dalam membuat sebuah goals yang akan digunakan untuk merancang materi, gunakan pengetahuan luasmu untuk membantu dalam membuat goals,kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal, kamu akan menerima input barupa name dari materi yang akan dibuat beserta level pendidikannya, tolong buatkan goals yang berkualitas

Respons hanya berupa soal dalam bentuk JSON dengan struktur:

{name} (jangan ubah ini cukup response dengan isi yang sama dengan yang dikirim)

{level_pendidikan} (jangan ubah ini cukup response dengan isi yang sama dengan yang dikirim)

{goals}
(dari name dan level_pendidikan yang diberikan bagian inilah yang harusnya kamu buatkan goals berdasarkan task yang diminta)

SELALU GUNAKAN FORMATTING HTML

Berikut adalah contoh respons yang benar:
{format}

ini ada refrensi untuk menambah pemahamanmu mengenai konteks yang diminta:
{context}

JANGAN merespons apapun selain soal berupa JSON
SELALU GUNAKAN FORMATTING HTML

dari instruksi tersebut lakukan task berikut

Task: 

"{task}" 

"""
)



class Goals(BaseModel):
    name: str = Field(
        description="jangan ubah ini cukup response dengan isi yang sama dengan yang dikirim")
    level_pendidikan: str = Field(
        description="jangan ubah ini cukup response dengan isi yang sama dengan yang dikirim")
    goals: str = Field(
        description="dari name dan level_pendidikan yang diberikan bagian inilah yang harusnya kamu buatkan goals berdasarkan task yang diminta")


parser = PydanticOutputParser(pydantic_object=Goals)


# check_option_prompt = PromptTemplate.from_template("dari input ini periksalah option dengan poin 1 sulit untuk dibedakan dengan option lainnya jika masih terlalu mudah silahkan ubah optionnya tapi ganti kontennya saja jangan ganti hal lainnya. struktur input yang diterima responselah dengan strukture yang sama /n {json_question_format}")



# format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template=template,
    input_variables=['task'],
    partial_variables= {
        "name": parser.get_format_instructions(),
        "level_pendidikan": parser.get_format_instructions(), 
        "goals": parser.get_format_instructions(),
        # "context": retriever
        },
)

input_prompts = [
    ("format", response_format_prompt),
]

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=prompt, pipeline_prompts=input_prompts
)

# chain = (
#     {"task": RunnablePassthrough()}
#     | pipeline_prompt 
#     | llm 
#     | {"json_question_format" : StrOutputParser()}
#     | check_option_prompt
#     | llm
#     | parser
# )

# setup retriever

# category_value = 'TWK'
# print(category_value)





# ANSWER_PROMPT = ChatPromptTemplate.from_template(
#     """anda adalah seorang asisten yang akan membantu dalam membuat soal, anda akan diberikan sebuah context untuk membantu dalam membuat soal.
#     buatlah soal berdasarkan task yang diberikan

#     context: {context}
#     task: "{question}"
#     Answer:
#     """
# )

chain = (
    {"context": retriever | RunnablePassthrough(), "task": RunnablePassthrough()}
    | pipeline_prompt 
    | llm 
    | parser
)



# Add typing for input
class InputPrompt(BaseModel):
    __root__: str
    # metadata:str

# class InputPrompt(TypedDict):
#     inputs:List[str]
#     metadata:str
    # info: Dict[str, Any]

# class Output(TypedDict):
#     output: chain.output_schema
#     info: Dict[str, Any]



chain = chain.with_types(input_type=InputPrompt)

