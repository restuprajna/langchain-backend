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
        "question": "Pada sebuah pertandingan sepak bola internasional, terjadi kericuhan antar suporter yang menyebabkan korban jiwa. Hal ini disebabkan oleh perbedaan pandangan dan sikap antarsuporter yang tidak mencerminkan sikap nasionalisme. Manakah dari pernyataan berikut yang merupakan implementasi sikap nasionalisme yang tepat dalam menyikapi peristiwa tersebut?",
        
        "answers": [
            {{
                "option": "A",
                "answer": "Menyalahkan pihak penyelenggara karena tidak bisa mengendalikan kericuhan.",
                "order": 1,
                "score": 0,
                "is_true": false
            }},
            {{    
                "option": "B",
                "answer": "Mengutuk tindakan suporter yang terlibat kericuhan dan menyerukan persatuan.",
                "order": 2,
                "score": 5,
                "is_true": true
            }},
            {{  
                "option": "C",
                "answer": "Menuntut pemerintah untuk membubarkan semua klub sepak bola yang suporternya terlibat kericuhan.",
                "order": 3,
                "score": 0,
                "is_true": false
            }},
            {{
                "option": "D",
                "answer": "Membenarkan tindakan suporter yang terlibat kericuhan karena didasari oleh rasa cinta tanah air.",
                "order": 4,
                "score": 0,
                "is_true": false
            }},
            {{       
                "option": "E",
                "answer": "Mengabaikan peristiwa tersebut karena dianggap bukan urusan pribadi.",
                "order": 5,
                "score": 0,
                "is_true": false
            }}
        ],                
                "explanation": "<b>Jawaban Yang Benar:</b> Mengutuk tindakan suporter yang terlibat kericuhan dan menyerukan persatuan.<br>Pernyataan ini menunjukkan sikap nasionalisme yang tepat karena mengedepankan persatuan dan kesatuan bangsa di atas perbedaan pandangan dan sikap.<br><br><b>Jawaban yang Salah:</b> <br>- Menyalahkan pihak penyelenggara karena tidak bisa mengendalikan kericuhan: Pernyataan ini tidak mencerminkan sikap nasionalisme karena menyalahkan pihak lain dan tidak mencari solusi yang konstruktif.<br><br>- Menuntut pemerintah untuk membubarkan semua klub sepak bola yang suporternya terlibat kericuhan: Pernyataan ini terlalu berlebihan dan tidak sesuai dengan prinsip nasionalisme yang mengedepankan persatuan dan kesatuan.<br><br>- Membenarkan tindakan suporter yang terlibat kericuhan karena didasari oleh rasa cinta tanah air: Pernyataan ini salah karena tindakan kekerasan tidak dapat dibenarkan dengan alasan apapun, termasuk rasa cinta tanah air.<br><br>- Mengabaikan peristiwa tersebut karena dianggap bukan urusan pribadi: Pernyataan ini menunjukkan sikap individualistik dan tidak mencerminkan sikap nasionalisme yang peduli terhadap kepentingan bangsa dan negara."
    }}
"""
)

response_format_prompt = PromptTemplate.from_template(StringPromptTemplate)



template = (
    """
Selalu ikuti instruksi berikut: kamu adalah asisten yang akan membantu membuat soal, gunakan pengetahuan luasmu untuk membantu dalam membuat soal,kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal, soal akan yang dibuat berdasarkan pada level bloom taksonomi yang diminta.

Anda akan menerima prompt dari pengguna berulang kali untuk membuat soal, jadi teruslah membuat soal yang baru. Terdapat 6 Level Taksonomi Bloom: Mengingat, Memahami, Menerapkan, Menganalisis, Mengevaluasi, dan Mencipta.

Jangan batasi kreativitas soal pada referensi, Anda bebas menggunakan pengetahuan Anda dalam membuat konteks. Pada penjelasan (explanation), jelaskan secara detail pada setiap opsi mengapa opsi tersebut benar dan mengapa opsi tersebut salah.

Respons hanya berupa soal dalam bentuk JSON dengan struktur:

{question} (Buatlah pertanyaan yang sesuai dengan level Taksonomi Bloom yang diminta)

{answers}
[option (Opsi hanya berisikan indikator dari opsi, yaitu dari A-E), answer (Berisikan konteks string opsi jawaban), order (1-5), score (Jika opsi benar, skor adalah 5. Jika opsi salah, skor adalah 0), is_true (true atau false)]

{explanation}
(Dari struktur answers[option, answer, order, score, is_true], berikan penjelasan untuk setiap answer. Selalu gunakan answer dari answers, jangan menggunakan indikator opsinya untuk merujuk pada opsi yang dimaksud. Jelaskan secara detail pada setiap answer mengapa answer tersebut benar atau mengapa answer tersebut salah, gunakan format sebagai berikut:

Jawaban Yang Benar: [Tulis isi dari answer tanpa indikator opsi] lalu pembahasan mengapa answer tersebut benar.

Jawaban yang salah: [Berisikan pembahasan masing-masing answer lainnya (hanya tulis isi dari answer tanpa indikator opsi)] dan mengapa answer tersebut salah.

SELALU GUNAKAN FORMATTING HTML

Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi. Opsi jawaban harus beragam dan logis, namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar.

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



class Question(BaseModel):
    question: str = Field(
        description="berisikan pertanyaan yang berkualitas berdasarkan task yang diminta")
    answers: list = Field(
        description="list yang berisikan struktur sebagai berikut: [option (option hanya berisikan indikator dari opsi yaitu dari A-E), answer(berisikan konteks string opsi jawaban), order (1-5) , score (if the option is correct the score is 5, if the option is wrong the score is 0, only one option is correct), is_true(true or false)]")
    explanation: str = Field(
        description="Dari struktur answers(option,answer, order, score, is_true) Berikan penjelasan untuk setiap answer, selalu gunakan answer dari answers jangan menggunakan indikator optionnya untuk merujuk pada opsi yang dimaksud, jelaskan secara detail pada tiap answer mengapa answer tersebut benar atau mengapa answer tersebut salah, gunakan format sebagai berikut: Jawaban Yang Benar: Answer(hanya tulis isi dari answer tanpa option indikator karena tidak penting) lalu pembahasan mengapa answer tersebut benar, lalu dilanjutkan dengan format : Jawaban yang salah: berisikan pembahasan masing-masing answer lainnya(hanya tulis isi dari answer tanpa option indikator karena tidak penting) dan mengapa answer tersebut salah, ALWAYS USE HTML FORMATTING")


parser = PydanticOutputParser(pydantic_object=Question)


# check_option_prompt = PromptTemplate.from_template("dari input ini periksalah option dengan poin 1 sulit untuk dibedakan dengan option lainnya jika masih terlalu mudah silahkan ubah optionnya tapi ganti kontennya saja jangan ganti hal lainnya. struktur input yang diterima responselah dengan strukture yang sama /n {json_question_format}")



# format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template=template,
    input_variables=['task'],
    partial_variables= {
        "question": parser.get_format_instructions(),
        "answers": parser.get_format_instructions(), 
        "explanation": parser.get_format_instructions(),
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

