import os

from langchain_community.retrievers import GoogleVertexAISearchRetriever
# from langchain_community.chat_models import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, StringPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
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
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}
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
llm = ChatGoogleGenerativeAI(temperature=1.0,
                            model=vertex_model, 
                            google_api_key=google_api, 
                            safety_settings=safety_settings_NONE)



# user_prompt = "Saya akan menguji kemampuan siswa saya tentang Menilai penguasaan pengetahuan dan kemampuan Pilar Negara yaitu Mampu membentuk karakter positif melalui pemahaman dan pengamalan nilai-nilai dalam pancasila, UUD 1945, NKRI, dan Bhinneka Tunggal Ika dalam pembelajaran Seleksi Kompetensi Dasar - Tes Wawasan Kebangsaan sub bagian Pilar Negara. Sekarang, saya ingin membuat soal untuk menguji kemampuan siswa tersebut. Buatlah soal baru berbentuk pilihan ganda dengan tujuan Pemahaman materi Pancasila sebagai Paradigma Pembangunan, dengan harapan Mengerti dan memahami tentang pilar negara serta Pancasila sebagai dasar, falsafah dan ideologi negara, Mengerti dan memahami konsep Pancasila sebagai ideologi terbuka dan sumber nilai, serta butir-butir pengamalan Pancasila berdasarkan Bloom Taksonomi 2001 level Menganalisis (C4). Gunakan peristiwa nyata yang terjadi baru-baru ini di Indonesia sebagai stimulus, kemudian buat soal berdasarkan kasus tersebut. Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi. Opsi jawaban harus beragam dan logis namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Hindari bahasa dan kalimat yang terlalu sederhana. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. Soal dibuat beserta pembahasan mengapa pilihan yang benar itu benar."

# ANSWER_PROMPT = ChatPromptTemplate.from_template(
#     """selalu ikuti instruksi berikut: kamu adalah seorang penguji dalam ujian Seleksi Kompetensi Bidang (SKB). Kamu bertugas untuk membuat soal SKB sesuai dengan kompetensi bidang masing-masing jabatan yang dipilih oleh calon ASN atau CPNS dalam seleksi akhir penerimaan Pegawai Negeri Sipil, dimana kamu hanya membuat soal, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal, soal yang dibuat berdasarkan pada level bloom taksonomi yang diminta. 
# ANSWER_PROMPT = ChatPromptTemplate.from_template(
#     """selalu ikuti instruksi berikut: kamu adalah seorang penguji dalam ujian Seleksi Kompetensi Bidang (SKB). Kamu bertugas untuk membuat soal SKB sesuai dengan kompetensi bidang masing-masing jabatan yang dipilih oleh calon ASN atau CPNS dalam seleksi akhir penerimaan Pegawai Negeri Sipil, dimana kamu hanya membuat soal, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal, soal yang dibuat berdasarkan pada level bloom taksonomi yang diminta. 

# kamu akan menerima user prompt yang sama berkali-kali untuk membuat soal, jadi teruslah membuat soal
# kamu akan menerima user prompt yang sama berkali-kali untuk membuat soal, jadi teruslah membuat soal

# Level Bloom Taksonomi: mengingat, memahami, menerapkan, menganalisis, mengevaluasi dan mencipta.
# Level Bloom Taksonomi: mengingat, memahami, menerapkan, menganalisis, mengevaluasi dan mencipta.

# saya sudah tambahkan file yang berisi kata kunci kognitif pada setiap level taksonomi bloom gunakan file tersebut sebagai tambahan referensi dalam membuat soal
# saya sudah tambahkan file yang berisi kata kunci kognitif pada setiap level taksonomi bloom gunakan file tersebut sebagai tambahan referensi dalam membuat soal

# Jangan batasi kreatifitas soal pada referensi, kamu bebas gunakan pengetahuanmu dalam membuat konteks.
# Jangan batasi kreatifitas soal pada referensi, kamu bebas gunakan pengetahuanmu dalam membuat konteks.

# respon hanya berupa soal dalam bentuk json dengan struktur:  
# -question, 
# -answers[option (A-E), answer, order (1-5) , score (0 or 5), is_true(true or false)],
# -explanation
# respon hanya berupa soal dalam bentuk json dengan struktur:  
# -question, 
# -answers[option (A-E), answer, order (1-5) , score (0 or 5), is_true(true or false)],
# -explanation

# Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi . Opsi jawaban harus beragam dan logical namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. Soal dibuat beserta pembahasan dari masing masing opsi mengapa opsi mendapat score tersebut
# Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi . Opsi jawaban harus beragam dan logical namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. Soal dibuat beserta pembahasan dari masing masing opsi mengapa opsi mendapat score tersebut

# JANGAN merespon apapun selain soal berupa JSON 
# JANGAN merespon apapun selain soal berupa JSON 


# dari instruksi tersebut lakukan task berikut

    
#     Task: "{task}"
#     Answer:
#     """
# )



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

# template = (
#     """selalu ikuti instruksi berikut: kamu adalah seorang penguji dalam ujian Seleksi Kompetensi Bidang (SKB). Kamu bertugas untuk membuat soal SKB sesuai dengan kompetensi bidang masing-masing jabatan yang dipilih oleh calon ASN atau CPNS dalam seleksi akhir penerimaan Pegawai Negeri Sipil, dimana kamu hanya membuat soal, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal, soal yang dibuat berdasarkan pada level bloom taksonomi yang diminta. 

# kamu akan menerima user prompt yang sama berkali-kali untuk membuat soal, jadi teruslah membuat soal

# Terdapat 6 Level Bloom Taksonomi: mengingat, memahami, menerapkan, menganalisis, mengevaluasi dan mencipta.

# Jangan batasi kreatifitas soal pada referensi, kamu bebas gunakan pengetahuanmu dalam membuat konteks.

# pada explanation jelaskan secara detail pada tiap opsi mengapa opsi tersebut benar dan mengapa opsi tersebut salah

# respon hanya berupa soal dalam bentuk json dengan struktur:  
# -{question}, 

# -{answers}[option (option hanya berisikan indikator dari opsi yaitu dari A-E), answer(berisikan konteks string opsi jawaban), order (1-5) , score (if the option is correct the score is 5, if the option is wrong the score is 0), is_true(true or false)],

# -{explanation} (Dari struktur answers(option,answer, order, score, is_true) Berikan penjelasan untuk setiap answer, selalu gunakan answer dari answers jangan menggunakan indikator optionnya untuk merujuk pada opsi yang dimaksud, jelaskan secara detail pada tiap answer mengapa answer tersebut benar atau mengapa answer tersebut salah, gunakan format sebagai berikut: Jawaban Yang Benar: Answer(hanya tulis isi dari answer tanpa option indikator karena tidak penting) lalu pembahasan mengapa answer tersebut benar, lalu dilanjutkan dengan format : Jawaban yang salah: berisikan pembahasan masing-masing answer lainnya(hanya tulis isi dari answer tanpa option indikator karena tidak penting) dan mengapa answer tersebut salah, ALWAYS USE HTML FORMATTING)

# Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi . Opsi jawaban harus beragam dan logical namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. 


# berikut adalah contoh response yang benar:
#     {format}
            

# JANGAN merespon apapun selain soal berupa JSON 
# ALWAYS USE HTML FORMATTING

# dari instruksi tersebut lakukan task berikut

    
#     Task: "{task}"
#     Answer:
#     """
# )

template = (
    """
Selalu ikuti instruksi berikut: kamu adalah ahli kebangsaan dan bahasa Indonesia tugasmu hanya membuat soal, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal, soal yang dibuat berdasarkan pada level bloom taksonomi yang diminta.

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

SELALU GUNAKAN FORMATTING HTML)

Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi. Opsi jawaban harus beragam dan logis, namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar.

Berikut adalah contoh respons yang benar:
{format}

JANGAN merespons apapun selain soal berupa JSON
SELALU GUNAKAN FORMATTING HTML

# dari instruksi tersebut lakukan task berikut

    
#     Task: "{task}"
#     Answer:
#     """
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

# template = (
#     """selalu ikuti instruksi berikut: kamu adalah seorang penguji dalam ujian Seleksi Kompetensi Bidang (SKB). Kamu bertugas untuk membuat soal SKB sesuai dengan kompetensi bidang masing-masing jabatan yang dipilih oleh calon ASN atau CPNS dalam seleksi akhir penerimaan Pegawai Negeri Sipil, dimana kamu hanya membuat soal, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal, soal yang dibuat berdasarkan pada level bloom taksonomi yang diminta. 

# kamu akan menerima user prompt yang sama berkali-kali untuk membuat soal, jadi teruslah membuat soal

# Terdapat 6 Level Bloom Taksonomi: mengingat, memahami, menerapkan, menganalisis, mengevaluasi dan mencipta.

# Jangan batasi kreatifitas soal pada referensi, kamu bebas gunakan pengetahuanmu dalam membuat konteks.

# pada explanation jelaskan secara detail pada tiap opsi mengapa opsi tersebut benar dan mengapa opsi tersebut salah

# respon hanya berupa soal dalam bentuk json dengan struktur:  
# -{question}, 

# -{answers}[option (option hanya berisikan indikator dari opsi yaitu dari A-E), answer(berisikan konteks string opsi jawaban), order (1-5) , score (if the option is correct the score is 5, if the option is wrong the score is 0), is_true(true or false)],

# -{explanation} (Dari struktur answers(option,answer, order, score, is_true) Berikan penjelasan untuk setiap answer, selalu gunakan answer dari answers jangan menggunakan indikator optionnya untuk merujuk pada opsi yang dimaksud, jelaskan secara detail pada tiap answer mengapa answer tersebut benar atau mengapa answer tersebut salah, gunakan format sebagai berikut: Jawaban Yang Benar: Answer(hanya tulis isi dari answer tanpa option indikator karena tidak penting) lalu pembahasan mengapa answer tersebut benar, lalu dilanjutkan dengan format : Jawaban yang salah: berisikan pembahasan masing-masing answer lainnya(hanya tulis isi dari answer tanpa option indikator karena tidak penting) dan mengapa answer tersebut salah, ALWAYS USE HTML FORMATTING)

# Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi . Opsi jawaban harus beragam dan logical namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. 


# berikut adalah contoh response yang benar:
#     {format}
            

# JANGAN merespon apapun selain soal berupa JSON 
# ALWAYS USE HTML FORMATTING

# dari instruksi tersebut lakukan task berikut

    
#     Task: "{task}"
#     Answer:
#     """
# )

template = (
    """
Selalu ikuti instruksi berikut: kamu adalah ahli kebangsaan dan bahasa Indonesia tugasmu hanya membuat soal, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal, soal yang dibuat berdasarkan pada level bloom taksonomi yang diminta.

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

SELALU GUNAKAN FORMATTING HTML)

Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi. Opsi jawaban harus beragam dan logis, namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar.

Berikut adalah contoh respons yang benar:
{format}

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


check_option_prompt = PromptTemplate.from_template("dari input ini periksalah option dengan poin 1 sulit untuk dibedakan dengan option lainnya jika masih terlalu mudah silahkan ubah optionnya tapi ganti kontennya saja jangan ganti hal lainnya. struktur input yang diterima responselah dengan strukture yang sama /n {json_question_format}")


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
    {"task": RunnablePassthrough()}
    | pipeline_prompt 
    | llm 
    | {"json_question_format" : StrOutputParser()}
    | check_option_prompt
    | llm
    | parser
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


