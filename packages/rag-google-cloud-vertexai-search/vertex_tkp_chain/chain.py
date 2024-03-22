import os

from langchain_community.retrievers import GoogleVertexAISearchRetriever
# from langchain_community.chat_models import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
    RetryOutputParser,
    ResponseSchema,
    StructuredOutputParser,
    RetryOutputParser
)
from dotenv import load_dotenv
# from langsmith import Client


load_dotenv

# client = Client()
# Get project, data store, and model type from env variables
# project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
# data_store_id = os.environ.get("DATA_STORE_ID")
# location = "global"
# model_type = os.environ.get("MODEL_TYPE")

# if not data_store_id:
#     raise ValueError(
#         "No value provided in env variable 'DATA_STORE_ID'. "
#         "A  data store is required to run this application."
#     )




# Set LLM and embeddings
# model = ChatVertexAI(model_name=model_type, temperature=0.0)

# google_api: str = os.environ["GOOGLE_API_KEY"]
# llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api)

# Create Vertex AI retriever
# retriever = GoogleVertexAISearchRetriever(
#     project_id=project_id, 
#     search_engine_id=data_store_id,
#     location=location
# )

# RAG prompt
# template = """Answer the question based only on the following context:
# {context}
# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# # RAG
# chain = (
#     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
#     | prompt
#     | llm
#     | StrOutputParser()
# )

safety_settings_NONE=safety_settings = {
    # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    # HarmCategory.HARM_CATEGORY_RUDE_OR_ABUSIVE: HarmBlockThreshold.BLOCK_NONE, 
    # HarmCategory.HARM_CATEGORY_PROFANITY: HarmBlockThreshold.BLOCK_NONE, 
    # HarmCategory.HARM_CATEGORY_ALCOHOL_TOBACCO_DRUGS: HarmBlockThreshold.BLOCK_NONE, 
    # HarmCategory.HARM_CATEGORY_OTHER: HarmBlockThreshold.BLOCK_NONE
}

google_api: str = os.environ["GOOGLE_API_KEY"]
vertex_model: str = os.environ["vertex_model"]

llm = ChatGoogleGenerativeAI(model=vertex_model, google_api_key=google_api, safety_settings=safety_settings_NONE)



# user_prompt = "Saya akan menguji kemampuan siswa saya tentang Menilai penguasaan pengetahuan dan kemampuan Pilar Negara yaitu Mampu membentuk karakter positif melalui pemahaman dan pengamalan nilai-nilai dalam pancasila, UUD 1945, NKRI, dan Bhinneka Tunggal Ika dalam pembelajaran Seleksi Kompetensi Dasar - Tes Wawasan Kebangsaan sub bagian Pilar Negara. Sekarang, saya ingin membuat soal untuk menguji kemampuan siswa tersebut. Buatlah soal baru berbentuk pilihan ganda dengan tujuan Pemahaman materi Pancasila sebagai Paradigma Pembangunan, dengan harapan Mengerti dan memahami tentang pilar negara serta Pancasila sebagai dasar, falsafah dan ideologi negara, Mengerti dan memahami konsep Pancasila sebagai ideologi terbuka dan sumber nilai, serta butir-butir pengamalan Pancasila berdasarkan Bloom Taksonomi 2001 level Menganalisis (C4). Gunakan peristiwa nyata yang terjadi baru-baru ini di Indonesia sebagai stimulus, kemudian buat soal berdasarkan kasus tersebut. Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi. Opsi jawaban harus beragam dan logis namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Hindari bahasa dan kalimat yang terlalu sederhana. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. Soal dibuat beserta pembahasan mengapa pilihan yang benar itu benar."

# ANSWER_PROMPT = ChatPromptTemplate.from_template(
#     """selalu ikuti instruksi berikut: kamu adalah seorang penguji dalam ujian Seleksi Kompetensi Bidang (SKB). Kamu bertugas untuk membuat soal SKB sesuai dengan kompetensi bidang masing-masing jabatan yang dipilih oleh calon ASN atau CPNS dalam seleksi akhir penerimaan Pegawai Negeri Sipil, dimana kamu hanya membuat soal, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal, soal yang dibuat berdasarkan pada level bloom taksonomi yang diminta. 

# kamu akan menerima user prompt yang sama berkali-kali untuk membuat soal, jadi teruslah membuat soal

# Level Bloom Taksonomi: mengingat, memahami, menerapkan, menganalisis, mengevaluasi dan mencipta.

# saya sudah tambahkan file yang berisi kata kunci kognitif pada setiap level taksonomi bloom gunakan file tersebut sebagai tambahan referensi dalam membuat soal

# Jangan batasi kreatifitas soal pada referensi, kamu bebas gunakan pengetahuanmu dalam membuat konteks.

# respon hanya berupa soal dalam bentuk json dengan struktur:  
# -question, 
# -answers[option (A-E), answer, order (1-5) , score (0 or 5), is_true(true or false)],
# -explanation

# Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi . Opsi jawaban harus beragam dan logical namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. Soal dibuat beserta pembahasan dari masing masing opsi mengapa opsi mendapat score tersebut

# JANGAN merespon apapun selain soal berupa JSON 


# dari instruksi tersebut lakukan task berikut

    
#     Task: "{task}"
#     Answer:
#     """
# )

template = (
     """selalu ikuti instruksi berikut: Anda adalah psikolog tugasmu hanya membuat soal untuk menilai karakter seseorang, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal. 

kamu akan menerima user prompt yang sama berkali-kali untuk membuat soal, jadi teruslah membuat soal

Level Bloom Taksonomi: mengingat, memahami, menerapkan, menganalisis, mengevaluasi dan mencipta.

saya sudah tambahkan file yang berisi kata kunci kognitif pada setiap level taksonomi bloom gunakan file tersebut sebagai tambahan referensi dalam membuat soal

Jangan batasi kreatifitas soal pada referensi, kamu bebas gunakan pengetahuanmu dalam membuat konteks.

jawaban di desain tanpa ada jawaban yang salah namun berikan skor dalam rentan 1 sampai 5, dan skor harus berbeda pada tiap opsi

respon hanya berupa soal dalam bentuk json dengan struktur:  
-{question}, 

-{answers}[option (option hanya berisikan indikator dari opsi yaitu dari A-E), answer(berisikan konteks string opsi jawaban), order (1-5) , score (berisikan nilain score tiap opsi dari rentang 1 sampai 5 dengan tiap opsi harus memiliki niali yang berbeda), is_true(true untuk opsi dengan or false)],

-{explanation} (tampilkan indikator soal dan isi jawaban/answer soal tersebut lalu jelaskan tiap opsi mengapa opsi tersebut benar dan mengapa opsi tersebut salah)

Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi . Opsi jawaban harus beragam dan logical namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. 

JANGAN merespon apapun selain soal berupa JSON 


dari instruksi tersebut lakukan task berikut

    
    Task: "{task}"
    Answer:
    """
)

# template = (
#     """cari kata dasar dalam kalimat berikut:
    
#     Task: "{task}"
#     Answer:
# """
# )



class Question(BaseModel):
    question: str = Field(description="the question to answer")
    answers: list = Field(
        description="list yang berisikan struktur sebagai berikut: [option (option hanya berisikan indikator dari opsi yaitu dari A-E), answer(berisikan konteks string opsi jawaban), order (1-5) , score (berisikan nilain score tiap opsi dari rentang 1 sampai 5 dengan tiap opsi harus memiliki niali yang berbeda), is_true(true untuk opsi dengan or false)]")
    explanation: str = Field(
        description="berisikan penjelasan pada tiap option mengapa opsi tersebut benar dan mengapa opsi tersebut salah, tiap sebelum menulis penjelaskan tulis ulang dulu opsi beserta answer yang dimaksud seperti format berikut: opsi.answer (salah atau benar) mengapa")


parser = PydanticOutputParser(pydantic_object=Question)

# retry_parser = RetryOutputParser.from_llm(parser=parser, llm=llm)
# retry_parser.parse_with_prompt(bad_response, prompt_value)
# response_schemas = [
#     ResponseSchema(
#         name="question", 
#         description="question to answer"),
#     ResponseSchema(
#         name="answers",
#         description="option to choose",
#     ),
#     ResponseSchema(
#         name="explanation",
#         description="explanation of the answer",
#     )
# ]
# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


# format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template=template,
    input_variables=['task'],
    partial_variables={"question": parser.get_format_instructions(), "answers": parser.get_format_instructions(), "explanation": parser.get_format_instructions()},
)
# chain = (
#     {"task": RunnablePassthrough()}
#     | ANSWER_PROMPT
#     | llm
#     | parser
# )
chain = (
    {"task": RunnablePassthrough()} |
    prompt | llm | parser
)

# main_chain = RunnableParallel(
#     completion=chain,
#     prompt_value=prompt
# ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

# ans = chain.invoke(user_prompt)

# retry_parser = RetryOutputParser.from_llm(
#     llm=chain.llm, 
#     parser=parser, 
#     max_retries=3
# )

# Add typing for input
class InputPrompt(BaseModel):
    __root__: str
    # input:InputPrompt
    # metadata: str


chain = chain.with_types(input_type=InputPrompt)

