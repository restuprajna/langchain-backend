import os

from langchain_community.retrievers import GoogleVertexAISearchRetriever
# from langchain_community.chat_models import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, StringPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
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

StringPromptTemplate=(
    """
    {{ 
        "question": "Di sebuah perusahaan multinasional, Anda bekerja sama dengan rekan kerja dari berbagai latar belakang budaya. Anda memperhatikan bahwa beberapa rekan kerja Anda lebih mudah beradaptasi dengan perbedaan budaya dibandingkan yang lain. Menurut Anda, apa faktor yang paling memengaruhi kemampuan seseorang untuk beradaptasi dengan lingkungan kerja yang majemuk?",
        "answers": [
            {{
                "option": "A",
                "answer": "Keterbukaan terhadap pengalaman baru",
                "order": 1,
                "score": 5,
                "is_true": true
            }},
            {{
                "option": "B",
                "answer": "Kemampuan untuk mengganti budaya sendiri dengan budaya yang baru",
                "order": 2,
                "score": 3,
                "is_true": false
            }},
            {{
                "option": "C",
                "answer": "Kemauan untuk mempelajari bahasa dan adat istiadat dari budaya lain",
                "order": 3,
                "score": 4,
                "is_true": false
            }},
            {{        
                "option": "D",
                "answer": "Kemampuan untuk menghindari konflik dengan rekan kerja dari budaya yang berbeda",
                "order": 4,
                "score": 2,
                "is_true": false
            }},
            {{
                "option": "E",
                "answer": "Keinginan untuk berteman hanya dengan orang-orang dari budaya sendiri",
                "order": 5,
                "score": 1,
                "is_true": false
            }},
        ],
                "explanation": "Jawaban yang Benar: Keterbukaan terhadap pengalaman baru <br> <b>(Score: 5)</b> <br> Seseorang yang terbuka terhadap pengalaman baru lebih cenderung menerima dan menghargai perbedaan budaya. Mereka bersedia untuk mencoba hal-hal baru dan belajar tentang budaya lain. <br> Jawaban yang Salah: <br> Kemampuan untuk mengganti budaya sendiri dengan budaya yang baru <br> <b>(Score: 3)</b> <br> Meskipun penting untuk menghormati budaya lain, seseorang tidak boleh mengganti budaya mereka sendiri. Menghargai keberagaman berarti menghargai semua budaya, termasuk budaya sendiri. <br> Kemauan untuk mempelajari bahasa dan adat istiadat dari budaya lain <br> <b>(Score: 4)</b> <br> Meskipun mempelajari bahasa dan adat istiadat dari budaya lain dapat membantu seseorang beradaptasi, hal tersebut bukanlah satu-satunya faktor yang menentukan. Keterbukaan terhadap pengalaman baru lebih penting karena mencakup kemauan untuk belajar dan menerima perbedaan. <br> Kemampuan untuk menghindari konflik dengan rekan kerja dari budaya yang berbeda <br> <b>(Score: 2)</b> <br> Menghindari konflik bukanlah cara yang efektif untuk beradaptasi dengan lingkungan kerja yang majemuk. Sebaliknya, seseorang harus berusaha untuk memahami dan mengatasi perbedaan budaya secara konstruktif. <br> Keinginan untuk berteman hanya dengan orang-orang dari budaya sendiri <br> <b>(Score: 1)</b> <br> Berteman hanya dengan orang-orang dari budaya sendiri menunjukkan kurangnya keterbukaan terhadap pengalaman baru dan merupakan penghalang untuk beradaptasi dengan lingkungan kerja yang majemuk.
    }}
    """
)

response_format_prompt = PromptTemplate.from_template(StringPromptTemplate)

template = (
    """selalu ikuti instruksi berikut: Anda adalah psikolog tugasmu adalah membantu dalam membuat soal untuk menilai karakter seseorang, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal. 

kamu akan menerima user prompt yang sama berkali-kali untuk membuat soal, jadi teruslah membuat soal

Jangan batasi kreatifitas soal pada referensi, kamu bebas gunakan pengetahuanmu dalam membuat konteks semakin panjang konteks soal semakin bagus. silahkan buat konteks soal dengan tokoh atau penggambaran keadaan di masyarat pokoknya silahkan dibuat sebebas mungkin ,Desain soal dengan High Order Thinking Skills (HOTS).

Soal yang diminta akan menyertakan level kognitif 6 level bloom taksonomi yaitu
Level Bloom Taksonomi: mengingat, memahami, menerapkan, menganalisis, mengevaluasi dan mencipta. Soal Dibuat dibuat menyesuaikan dengan level bloom yang diminta

puntuk option di desain tanpa ada jawaban yang salah namun berikan score dalam yang harus berbeda/unique pada satu opsi deengan opsi yang lain, dengan rentang nilai 1-5, dan tiap skor pada masing-masing opsi haruslah berbeda/unique satu sama lainnya.

pastikan respon yang dibuat harus selalu mengikuti kriteria dan struktur sebagai berikut:  
-{question} (susunlah soal dengan cara berikut:  pertama, buatlah scenario cerita dengan panjang minimal tiga kalimat silahkan buat cerita yang kreatif semakin panjang cerita pada soal semakin bagus, lalu yang keduakedua dari cerita tersebut tanyakan sikap yang seseorang terhadap cerita tersebut)

-{answers}[option (option hanya berisikan indikator dari opsi yaitu dari A-E), answer(berisikan konteks string opsi jawaban), order (1-5) , score (berisikan nilai score yang harus berbeda/unique pada satu opsi dengan opsi yang lain dengan rentang nilai terendah 1 dan nilai terginggi 5, dan ingatlah tiap opsi harus memiliki nilai yang berbeda/unique satu sama lainnya), is_true(true untuk opsi dengan or false)],

-{explanation} (Dari struktur answers[option,answer, order, score, is_true] berikan penjelasan untuk setiap answer, selalu gunakan answer dari answers jangan menggunakan indikator optionnya untuk merujuk pada answer yang dimaksud, jelaskan secara detail pada tiap answer mengapa answer tersebut benar atau mengapa answer tersebut kurang benar,gunakan format sebagai berikut: Jawaban Yang Benar: Answer(hanya tulis isi dari answer tanpa option indikator karena tidak penting) tulis dengan score(score-nya) lalu pembahasan mengapa answer tersebut benar, lalu dilanjutkan dengan format : Jawaban yang salah: berisikan pembahasan masing-masing answer lainnya(hanya tulis isi dari answer tanpa option indikator karena tidak penting) tulis dengan score(scorenya) dan mengapa answer tersebut mendapat score yang demikian. Jangan gunakan formatting lain selain HTML)

Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi . Opsi jawaban harus beragam dan logical namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. 


berikut adalah contoh response yang benar:
    {format}


ALWAYS USE HTML TAG FOR FORMATTING

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
    question: str = Field(
        description="berisi soal HIGH ORDER THINKING SKILLS yang harus sesuai kriteria yang diminta, pastikan soal memiliki konteks yang panjang untuk menambah kompleksitas soal, semakin panjang konteks soal semakin baik")
    answers: list = Field(
        description="list yang berisikan option jawaban yang harus sesuai kriteria yang diminta")
    explanation: str = Field(
    description="deskripsi jawaban yang harus sesuai kriteria yang diminta tanpa menulis A/B/C/D/E dari optionya, ALWAYS USE HTML TAG FOR FORMATTING, selalu diurutkan dari skor tertinggi ke terendah")


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

input_prompts = [
    ("format", response_format_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=prompt, pipeline_prompts=input_prompts
)
# chain = (
#     {"task": RunnablePassthrough()}
#     | ANSWER_PROMPT
#     | llm
#     | parser
# )
chain = (
    {"task": RunnablePassthrough()} |
    pipeline_prompt | llm | parser
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

