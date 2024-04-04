import os

from langchain_community.retrievers import GoogleVertexAISearchRetriever
# from langchain_community.chat_models import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser

from langchain_openai import ChatOpenAI
# from langsmith import Client

from dotenv import load_dotenv

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


openai_api: str = os.environ["OPENAI_API_KEY"]
gpt_model: str = os.environ["gpt_model"]
llm = ChatOpenAI(model=gpt_model)



# user_prompt = "Saya akan menguji kemampuan siswa saya tentang Menilai penguasaan pengetahuan dan kemampuan Pilar Negara yaitu Mampu membentuk karakter positif melalui pemahaman dan pengamalan nilai-nilai dalam pancasila, UUD 1945, NKRI, dan Bhinneka Tunggal Ika dalam pembelajaran Seleksi Kompetensi Dasar - Tes Wawasan Kebangsaan sub bagian Pilar Negara. Sekarang, saya ingin membuat soal untuk menguji kemampuan siswa tersebut. Buatlah soal baru berbentuk pilihan ganda dengan tujuan Pemahaman materi Pancasila sebagai Paradigma Pembangunan, dengan harapan Mengerti dan memahami tentang pilar negara serta Pancasila sebagai dasar, falsafah dan ideologi negara, Mengerti dan memahami konsep Pancasila sebagai ideologi terbuka dan sumber nilai, serta butir-butir pengamalan Pancasila berdasarkan Bloom Taksonomi 2001 level Menganalisis (C4). Gunakan peristiwa nyata yang terjadi baru-baru ini di Indonesia sebagai stimulus, kemudian buat soal berdasarkan kasus tersebut. Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi. Opsi jawaban harus beragam dan logis namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Hindari bahasa dan kalimat yang terlalu sederhana. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. Soal dibuat beserta pembahasan mengapa pilihan yang benar itu benar."

template = (
    """selalu ikuti instruksi berikut: Anda adalah psikolog tugasmu hanya membuat soal untuk menilai karakter seseorang, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal. 

kamu akan menerima user prompt yang sama berkali-kali untuk membuat soal, jadi teruslah membuat soal

Level Bloom Taksonomi: mengingat, memahami, menerapkan, menganalisis, mengevaluasi dan mencipta.

Jangan batasi kreatifitas soal pada referensi, kamu bebas gunakan pengetahuanmu dalam membuat konteks.

option di desain tanpa ada jawaban yang salah namun berikan score dalam yang harus berbeda/unique pada satu opsi deengan opsi yang lain, dengan rentang nilai 1-5, dan tiap skor pada masing-masing opsi haruslah berbeda/unique satu sama lainnya.

pastikan respon yang dibuat harus selalu mengikuti kriteria dan struktur sebagai berikut:  
-{question} (susunlah soal dengan cara berikut:  pertama, buatlah scenario atau sebuah cerita dengan panjang minimal dua kalimat silahkan buat cerita yang sekreatif mungkin ,kedua dari cerita tersebut tanyakan sikap yang seseorang terhadap cerita tersebut)

-{answers}[option (option hanya berisikan indikator dari opsi yaitu dari A-E), answer(berisikan konteks string opsi jawaban), order (1-5) , score (berisikan nilai score yang harus berbeda/unique pada satu opsi dengan opsi yang lain dengan rentang nilai terendah 1 dan nilai terginggi 5, dan ingatlah tiap opsi harus memiliki nilai yang berbeda/unique satu sama lainnya), is_true(true untuk opsi dengan or false)],

-{explanation} (Dari struktur answers[option,answer, order, score, is_true] berikan penjelasan untuk setiap answer, selalu gunakan answer dari answers jangan menggunakan indikator optionnya untuk merujuk pada answer yang dimaksud, jelaskan secara detail pada tiap answer mengapa answer tersebut benar atau mengapa answer tersebut kurang benar,gunakan format sebagai berikut: Jawaban Yang Benar: Answer(hanya tulis isi dari answer tanpa option indikator karena tidak penting) tulis dengan score(score-nya) lalu pembahasan mengapa answer tersebut benar, lalu dilanjutkan dengan format : Jawaban yang salah: berisikan pembahasan masing-masing answer lainnya(hanya tulis isi dari answer tanpa option indikator karena tidak penting) tulis dengan score(scorenya) dan mengapa answer tersebut mendapat score yang demikian. Jangan gunakan formatting lain selain HTML)

Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi . Opsi jawaban harus beragam dan logical namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. 

JANGAN merespon apapun selain soal berupa JSON 
ALWAYS USE HTML TAG FOR FORMATTING

dari instruksi tersebut lakukan task berikut

    
    Task: "{task}"
    Answer:
    """
)


class Question(BaseModel):
    question: str = Field(
        description="berisi pertanyaan yang harus sesuai kriteria yang diminta")
    answers: list = Field(
        description="list yang berisikan option jawaban yang harus sesuai kriteria yang diminta")
    explanation: str = Field(
    description="deskripsi jawaban yang harus sesuai kriteria yang diminta tanpa menulis A/B/C/D/E dari optionya, ALWAYS USE HTML TAG FOR FORMATTING")


parser = PydanticOutputParser(pydantic_object=Question)

prompt = PromptTemplate(
    template=template,
    input_variables=['task'],
    partial_variables={"question": parser.get_format_instructions(), "answers": parser.get_format_instructions(), "explanation": parser.get_format_instructions()},
)

chain = (
    {"task": RunnablePassthrough()} |
    prompt | llm | parser
)


# ans = chain.invoke(user_prompt)



# Add typing for input
class InputPrompt(BaseModel):
    __root__: str
    # input:InputPrompt
    # metadata: str


chain = chain.with_types(input_type=InputPrompt)
