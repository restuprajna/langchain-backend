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

llm = ChatGoogleGenerativeAI(temperature= 1.0 ,
                            model=vertex_model, 
                            google_api_key=google_api, 
                            safety_settings=safety_settings_NONE,
                            )



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

# StringPromptTemplate=(
#     """
#     {{ 
#         "question": "Di sebuah perusahaan multinasional, Anda bekerja sama dengan rekan kerja dari berbagai latar belakang budaya. Anda memperhatikan bahwa beberapa rekan kerja Anda lebih mudah beradaptasi dengan perbedaan budaya dibandingkan yang lain. Menurut Anda, apa faktor yang paling memengaruhi kemampuan seseorang untuk beradaptasi dengan lingkungan kerja yang majemuk?",
#         "answers": [
#             {{
#                 "option": "A",
#                 "answer": "Keterbukaan terhadap pengalaman baru",
#                 "order": 1,
#                 "score": 5,
#                 "is_true": true
#             }},
#             {{
#                 "option": "B",
#                 "answer": "Kemampuan untuk mengganti budaya sendiri dengan budaya yang baru",
#                 "order": 2,
#                 "score": 3,
#                 "is_true": false
#             }},
#             {{
#                 "option": "C",
#                 "answer": "Kemauan untuk mempelajari bahasa dan adat istiadat dari budaya lain",
#                 "order": 3,
#                 "score": 4,
#                 "is_true": false
#             }},
#             {{        
#                 "option": "D",
#                 "answer": "Kemampuan untuk menghindari konflik dengan rekan kerja dari budaya yang berbeda",
#                 "order": 4,
#                 "score": 2,
#                 "is_true": false
#             }},
#             {{
#                 "option": "E",
#                 "answer": "Keinginan untuk berteman hanya dengan orang-orang dari budaya sendiri",
#                 "order": 5,
#                 "score": 1,
#                 "is_true": false
#             }},
#         ],
#                 "explanation": "Jawaban yang Benar: Keterbukaan terhadap pengalaman baru <br> <b>(Score: 5)</b> <br> Seseorang yang terbuka terhadap pengalaman baru lebih cenderung menerima dan menghargai perbedaan budaya. Mereka bersedia untuk mencoba hal-hal baru dan belajar tentang budaya lain. <br> Jawaban yang Salah: <br> Kemampuan untuk mengganti budaya sendiri dengan budaya yang baru <br> <b>(Score: 3)</b> <br> Meskipun penting untuk menghormati budaya lain, seseorang tidak boleh mengganti budaya mereka sendiri. Menghargai keberagaman berarti menghargai semua budaya, termasuk budaya sendiri. <br> Kemauan untuk mempelajari bahasa dan adat istiadat dari budaya lain <br> <b>(Score: 4)</b> <br> Meskipun mempelajari bahasa dan adat istiadat dari budaya lain dapat membantu seseorang beradaptasi, hal tersebut bukanlah satu-satunya faktor yang menentukan. Keterbukaan terhadap pengalaman baru lebih penting karena mencakup kemauan untuk belajar dan menerima perbedaan. <br> Kemampuan untuk menghindari konflik dengan rekan kerja dari budaya yang berbeda <br> <b>(Score: 2)</b> <br> Menghindari konflik bukanlah cara yang efektif untuk beradaptasi dengan lingkungan kerja yang majemuk. Sebaliknya, seseorang harus berusaha untuk memahami dan mengatasi perbedaan budaya secara konstruktif. <br> Keinginan untuk berteman hanya dengan orang-orang dari budaya sendiri <br> <b>(Score: 1)</b> <br> Berteman hanya dengan orang-orang dari budaya sendiri menunjukkan kurangnya keterbukaan terhadap pengalaman baru dan merupakan penghalang untuk beradaptasi dengan lingkungan kerja yang majemuk.
#     }}
#     """
# )



StringPromptTemplate = (
    """
{{
            "question": "Dalam rangka meningkatkan kualitas pelayanan yang diberikan, sebuah perusahaan yang bergerak di bidang perhotelan melakukan survei kepada para tamunya. Salah satu pertanyaan dalam survei tersebut menanyakan pendapat tamu tentang fasilitas yang disediakan oleh pihak hotel. Salah satu tamu yang mengisi survei tersebut memberikan komentar bahwa ia merasa kurang puas dengan fasilitas sarapan pagi yang disediakan oleh pihak hotel. Menurut Anda, sebagai seorang karyawan di bagian pelayanan tamu, sikap yang tepat untuk menanggapi komentar tamu tersebut adalah...",
            "answers": [
                {{
                    "option": "A",
                    "answer": "Menjelaskan kepada tamu bahwa fasilitas sarapan pagi sudah sesuai dengan standar yang ditetapkan oleh manajemen hotel.",
                    "order": 1,
                    "score": 2,
                    "is_true": false
                }},
                {{
                    "option": "B",
                    "answer": "Meminta maaf atas ketidakpuasan tamu dan berjanji akan menyampaikan keluhan tersebut kepada manajemen hotel.",
                    "order": 2,
                    "score": 4,
                    "is_true": false
                }},
                {{
                    "option": "C",
                    "answer": "Mencoba membujuk tamu untuk mengubah pendapatnya tentang fasilitas sarapan pagi yang disediakan.",
                    "order": 3,
                    "score": 1,
                    "is_true": false
                }},
                {{
                    "option": "D",
                    "answer": "Menjelaskan kepada tamu bahwa fasilitas sarapan pagi yang disediakan sudah sesuai dengan harga yang dibayarkan tamu.",
                    "order": 4,
                    "score": 3,
                    "is_true": false
                }},
                {{
                    "option": "E",
                    "answer": "Menyampaikan terima kasih atas masukan dari tamu dan berjanji akan menindaklanjutinya dengan pihak terkait.",
                    "order": 5,
                    "score": 5,
                    "is_true": true
                }}
            ],
            "explanation": "Jawaban yang Benar: Menyampaikan terima kasih atas masukan dari tamu dan berjanji akan menindaklanjutinya dengan pihak terkait. <br> <b>(Score: 5)</b> <br> Sikap ini menunjukkan bahwa karyawan menghargai masukan dari tamu dan bersedia untuk mengambil tindakan untuk meningkatkan kualitas pelayanan. <br> Jawaban yang Salah: <br> Menjelaskan kepada tamu bahwa fasilitas sarapan pagi sudah sesuai dengan standar yang ditetapkan oleh manajemen hotel. <br> <b>(Score: 2)</b> <br> Sikap ini menunjukkan bahwa karyawan tidak mau menerima kritik dan tidak berusaha untuk meningkatkan kualitas pelayanan. <br> Meminta maaf atas ketidakpuasan tamu dan berjanji akan menyampaikan keluhan tersebut kepada manajemen hotel. <br> <b>(Score: 4)</b> <br> Sikap ini menunjukkan bahwa karyawan peduli dengan kepuasan tamu, namun tidak menunjukkan bahwa mereka akan mengambil tindakan untuk meningkatkan kualitas pelayanan. <br> Mencoba membujuk tamu untuk mengubah pendapatnya tentang fasilitas sarapan pagi yang disediakan. <br> <b>(Score: 1)</b> <br> Sikap ini menunjukkan bahwa karyawan tidak menghargai pendapat tamu dan tidak berusaha untuk meningkatkan kualitas pelayanan. <br> Menjelaskan kepada tamu bahwa fasilitas sarapan pagi yang disediakan sudah sesuai dengan harga yang dibayarkan tamu. <br> <b>(Score: 3)</b> <br> Sikap ini menunjukkan bahwa karyawan lebih mementingkan keuntungan hotel daripada kepuasan tamu."
        }}
"""
)

response_format_prompt = PromptTemplate.from_template(StringPromptTemplate)

# template = (
#     """selalu ikuti instruksi berikut: Anda adalah psikolog tugasmu adalah membantu dalam membuat soal untuk menilai karakter seseorang, kamu tidak bisa langsung berkomunikasi dengan pengguna karena kamu hanya bisa merespon dengan soal. 

# kamu akan menerima user prompt yang sama berkali-kali untuk membuat soal, jadi teruslah membuat soal

# Hasilkan pertanyaan yang kreatif dan tingkat tinggi. Pastikan pertanyaan yang dihasilkan memerlukan pemikiran tingkat tinggi, termasuk analisis, sintesis, dan evaluasi. Pertimbangkan berbagai sudut pandang dan kemungkinan solusi yang kompleks. Jangan batasi diri pada format atau pendekatan tertentu, biarkan kreativitas Anda mengalir dalam pembuatan pertanyaan.

# Soal yang diminta akan menyertakan level kognitif 6 level bloom taksonomi yaitu
# Level Bloom Taksonomi: mengingat, memahami, menerapkan, menganalisis, mengevaluasi dan mencipta. Soal Dibuat dibuat menyesuaikan dengan level bloom yang diminta

# Lalu Generate opsi yang berkualitas tinggi untuk pertanyaan. Opsi yang dihasilkan harus memiliki kualitas yang tinggi dan tidak ada yang salah secara langsung. Namun, pastikan untuk membuat opsi yang sangat mirip satu sama lain, sehingga mempersulit pemilih dalam menentukan opsi yang paling tepat. Setiap opsi harus memiliki skor yang berbeda dari 1 hingga 5, dengan 5 menunjukkan tingkat kesesuaian yang paling tinggi dan 1 menunjukkan tingkat kesesuaian yang paling rendah. Pastikan untuk mempertimbangkan keunikan dan relevansi setiap opsi terhadap pertanyaan yang diberikan."

# untuk option di desain tanpa ada jawaban yang salah namun berikan score dalam yang harus berbeda/unique pada satu opsi deengan opsi yang lain, dengan rentang nilai 1-5, dan tiap skor pada masing-masing opsi haruslah berbeda/unique satu sama lainnya.

# pastikan respon yang dibuat harus selalu mengikuti kriteria dan struktur sebagai berikut:  
# -{question} (susunlah soal dengan cara berikut:  pertama, buatlah scenario cerita dengan panjang minimal tiga kalimat silahkan buat cerita yang kreatif, soal tidak boleh singkat atau pendek semakin panjang cerita pada soal semakin bagus, lalu yang keduakedua dari cerita tersebut tanyakan sikap yang seseorang terhadap cerita tersebut)

# -{answers}[option (option hanya berisikan indikator dari opsi yaitu dari A-E), answer(berisikan konteks string opsi jawaban), order (1-5) , score (berisikan nilai score yang harus berbeda/unique pada satu opsi dengan opsi yang lain dengan rentang nilai terendah 1 dan nilai terginggi 5, dan ingatlah tiap opsi harus memiliki nilai yang berbeda/unique satu sama lainnya), is_true(true untuk opsi dengan or false)],

# -{explanation} (Dari struktur answers[option,answer, order, score, is_true] berikan penjelasan untuk setiap answer, selalu gunakan answer dari answers jangan menggunakan indikator optionnya untuk merujuk pada answer yang dimaksud, jelaskan secara detail pada tiap answer mengapa answer tersebut benar atau mengapa answer tersebut kurang benar,gunakan format sebagai berikut: Jawaban Yang Benar: Answer(hanya tulis isi dari answer tanpa option indikator karena tidak penting) tulis dengan score(score-nya) lalu pembahasan mengapa answer tersebut benar, lalu dilanjutkan dengan format : Jawaban yang salah: berisikan pembahasan masing-masing answer lainnya(hanya tulis isi dari answer tanpa option indikator karena tidak penting) tulis dengan score(scorenya) dan mengapa answer tersebut mendapat score yang demikian. Jangan gunakan formatting lain selain HTML)

# Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi . Opsi jawaban harus beragam dan logical namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar. 


# berikut adalah contoh response yang benar:
#     {format}


# ALWAYS USE HTML TAG FOR FORMATTING

# dari instruksi tersebut lakukan task berikut

    
#     Task: "{task}"
#     Answer:
#     """
# )


template = (
    """
Selalu ikuti instruksi berikut: Anda adalah psikolog. Tugas Anda adalah membantu membuat soal untuk menilai karakter seseorang. Anda tidak dapat berkomunikasi langsung dengan pengguna karena Anda hanya dapat merespons dengan membuat soal. Anda akan menerima prompt dari pengguna berulang kali untuk membuat soal, jadi teruslah membuat soal yang baru.

Hasilkan pertanyaan yang kreatif dan tingkat tinggi. Pastikan pertanyaan yang dihasilkan memerlukan pemikiran tingkat tinggi, termasuk analisis, sintesis, dan evaluasi. Pertimbangkan berbagai sudut pandang dan kemungkinan solusi yang kompleks. Jangan batasi diri pada format atau pendekatan tertentu, biarkan kreativitas Anda mengalir dalam pembuatan pertanyaan.

Soal yang diminta akan menyertakan level kognitif 6 level Taksonomi Bloom, yaitu: Mengingat, Memahami, Menerapkan, Menganalisis, Mengevaluasi, dan Mencipta. Soal dibuat menyesuaikan dengan level Taksonomi Bloom yang diminta.

Lalu, hasilkan opsi pilihan ganda yang berkualitas tinggi minimal 15 kata untuk pertanyaan tersebut. Opsi yang dihasilkan harus memiliki kualitas yang tinggi dan menimbulkan dilema etis dalam mengambil keputusan sehingga siswa harus kritis untuk menjawab soal dengan opsi tersebut. Namun, pastikan untuk membuat opsi yang sangat mirip satu sama lain, sehingga mempersulit pemilih dalam menentukan opsi yang paling tepat.

Setiap opsi harus memiliki skor yang berbeda dari 1 hingga 5, dengan 5 menunjukkan tingkat kesesuaian yang paling tinggi dan 1 menunjukkan tingkat kesesuaian yang paling rendah. Pastikan untuk mempertimbangkan relevansi setiap opsi terhadap pertanyaan yang diberikan. Opsi yang dihasilkan harus memiliki kualitas yang tinggi dan menimbulkan dilema etis dalam mengambil keputusan sehingga siswa harus kritis untuk menjawab soal dengan opsi tersebut

Untuk opsi, desain Opsi yang dihasilkan harus memiliki kualitas yang tinggi dan menimbulkan dilema etis dalam mengambil keputusan sehingga siswa harus kritis untuk menjawab soal dengan opsi tersebut minimal 15 kata, namun berikan skor yang harus berbeda/unik pada satu opsi dengan opsi yang lain, dengan rentang nilai 1-5, dan tiap skor pada masing-masing opsi haruslah berbeda/unik satu sama lainnya.

Pastikan respons yang dibuat selalu mengikuti kriteria dan struktur sebagai berikut:

{question} (Susunlah soal dengan cara berikut: Pertama, buatlah skenario cerita sebagai konteks dengan panjang minimal 150 kata sampai maksimal 200 kata. Silahkan buat cerita yang kreatif. cerita bisa berupa latar belakang seorang tokoh, menceritakan tokoh fiksi, penggambaran sebuah keadaan yang detail kondisinya harus jelas jangan buat konteks yang terlalu umum atau general. Semakin panjang cerita pada soal semakin bagus. Kedua, dari cerita tersebut, tanyakan sikap yang seseorang terhadap cerita tersebut.)

berikut adalah beberapa contoh standar response soal yang diinginkan, Untuk membuat pertanyaan yang memiliki konteks yang lebih panjang seperti yang Anda inginkan, Anda dapat menyertakan detail tambahan tentang situasi atau latar belakang karakter dalam pertanyaan Anda. Berikut adalah contoh cara Anda dapat menyempurnakan pertanyaan Anda:
    1. Pak Samuel memiliki anak semata wayang bernama Aulia. Suatu hari, diketahui anak Pak Samuel adalah seseorang yang baru lulus di SMK swasta dengan program studi akuntansi. Saat ini Aulia bekerja sebagai seorang akuntan di suatu perusahaan perhiasan emas. Tuntutan atasannya tentu mengharuskan Aulia bekerja efektif sesuai standar perusahaan. Sementara Aulia baru lulus dari sekolahnya. Agar dapat bekerja optimal sesuai tuntutan atasannya, apa yang sebaiknya dilakukan Aulia?

    2. Seperti biasanya, Anda selalu disibukkan dengan rutinitas pekerjaan Anda sebagai costumer service di Bank A. Pada siang ini, Anda sangat disibukkan dengan menginput data nasabah karena lonjakan jumlah nasabah bulan ini. Ketika sedang menginput, ada ibu-ibu yang sudah tua mendatangi Anda. Ibu tersebut menanyakan terkait prosedur pencairan dana pensiunnya yang akan digunakan untuk membiayai anaknya yang sedang dirawat di rumah sakit. Ibu tersebut sangat membutuhkan uang tersebut untuk membayar semua biaya tersebut. Anda menjelaskan terkait prosedur pencairan dana tersebut dengan bahasa yang sederhana dan mudah dipahami. Akan tetapi, ibu tersebut kebingungan dan mengatakan bahwa dia tetap belum mengerti terkait prosedur tersebut. Hal yang akan Anda lakukan selanjutnya adalah ...

    3. Anda adalah seorang guru wanita yang sudah bekerja selama lima tahun. Sehari-hari Anda ditugaskan untuk mengajar murid SMP. Suatu hari, Anda mendapat tawaran untuk mengikuti program pertukaran guru selama sebulan ke sebuah sekolah di Aceh. Karena Anda beragama islam, Anda tertarik untuk mengikuti program tersebut. Di hari pertama Anda mulai mengajar di sana, Anda memperhatikan bahwa setiap Wanita di sekolah tersebut, baik guru, maupun murid, menggunakan jilbab. Anda sendiri belum mengenakan jilbab. Apa tindakan yang sebaiknya Anda lakukan?

    4. Setiap perusahaan tentu menuntut pekerja perusahaan tersebut untuk bekerja secara profesional. Salah satunya yaitu setiap pekerja harus beraktivitas sesuai dengan kewajiban dan tanggung jawab masing-masing pekerja yang telah ditetapkan pada kontrak kinerja. Di lingkungan tempat saya bekerja ada peraturan yang mengatur tentang larangan menggunakan gadget untuk bermain selama bekerja. Namun pada kenyataannya atasan dan rekan kerja saya kerap kali melanggar peraturan tersebut sehinga bertentangan dengan hitam di atas putih pada perusahaan. Sikap saya dalam hal tersebut adalah ...

JANGAN hasilkan soal yang pendek dan minim konteks seperti beberapa soal berikut:
    1. Ketika Anda bekerja di lingkungan kerja yang majemuk, di mana Anda berinteraksi dengan rekan kerja dari berbagai latar belakang budaya, menurut Anda apa hal terpenting yang harus dilakukan untuk membangun hubungan kerja yang harmonis?
    2. Dalam sebuah rapat kerja, seorang anggota tim mengutarakan pendapat yang bertentangan dengan pandangan kelompok. Sikap Anda terhadap pendapat tersebut adalah ...
    3. Dalam menghadapi perbedaan pendapat yang terjadi di masyarakat, sikap yang tepat untuk menghindari konflik dan perpecahan adalah...
    4. Dalam sebuah tim kerja yang terdiri dari individu dengan latar belakang budaya yang beragam, saya percaya bahwa kemampuan terpenting untuk beradaptasi secara efektif adalah..
    5. Di sebuah pusat perbelanjaan yang ramai, Anda menyaksikan seorang wanita tua kesulitan membawa belanjaannya yang banyak dan berat. Sebagai karyawan pusat perbelanjaan, sikap yang tepat untuk Anda lakukan adalah..


{answers}
[option (Opsi hanya berisikan indikator dari opsi, yaitu dari A-E minimal 15 kata untuk masing-masing opsi), answer (Berisikan konteks string opsi jawaban), order (1-5), score (Berisikan nilai skor yang harus berbeda/unik pada satu opsi dengan opsi yang lain dengan rentang nilai terendah 1 dan nilai tertinggi 5, dan ingatlah tiap Opsi yang dihasilkan harus memiliki kualitas yang tinggi dan menimbulkan dilema etis dalam mengambil keputusan sehingga siswa harus kritis untuk menjawab soal dengan opsi tersebut), is_true (true untuk opsi dengan skor tertinggi, false untuk lainnya)]

Berikut adalah kriteria dari masing masing score yang diberikan pada opsi, opsi yang dibuat diharapkan sulit untuk dibedakan scorenya jika dibaca sekilas. diharapkan soal dengan score 5 dan score 1 tidak memiliki perbedaan yang sangat drastis sehingga mudah sekali dibedakan sehingga menimbulkan dilema ketika memilih jawaban yang benar dari opsinya: 
question : Saya sedang mengerjakan laporan tugas semester yang akan dikumpulkan besok pagi. Tiba-tiba sahabat saya datang dengan wajah sedih dan ingin curhat pada saya. Sikap saya adalah...

-Score 5 : merangkum solusi yang paling etis dan paling memperhatikan keadilan, kebaikan umum, dan prinsip-prinsip moral yang tinggi. Jawaban ini harus menunjukkan pemahaman mendalam tentang situasi yang dihadapi dan menawarkan solusi yang paling sesuai dengan standar etika yang tinggi. (B. Memberikan tanggapan yang wajar sambil tetap mengerjakan laporan saya)

-Score 4 :  harus masih mengutamakan prinsip-prinsip moral dan etika, namun mungkin memiliki beberapa kompromi atau implikasi yang tidak terlalu jelas. Jawaban ini mungkin menawarkan pendekatan yang lebih praktis atau memperhitungkan faktor-faktor lain yang tidak sepenuhnya bersifat moral, tetapi tetap mempertahankan tingkat integritas moral yang tinggi (E. Dengan menyesal menolak mende ngarkan keluhannya)

-Score 3 : mungkin menunjukkan adanya dilema antara kepentingan individu dan kepentingan kelompok atau antara prinsip-prinsip etis yang berbeda. Jawaban ini bisa mencakup pertimbangan etika yang kompleks namun juga mempertimbangkan faktor-faktor praktis atau pribadi yang mungkin mempengaruhi pengambilan keputusan (C. Meneruskan mengerjakan laporan tanpa mempedulikan teman saya)

-Score 2 : Opsi dengan skor 2 mungkin mencerminkan pendekatan yang lebih pragmatis atau mengutamakan kepentingan pribadi atau kelompok atas keadilan atau moralitas umum. Jawaban ini mungkin menunjukkan kecenderungan untuk mengabaikan beberapa prinsip etika dalam mendukung kepentingan tertentu atau dalam situasi di mana pertimbangan etika tidak diutamakan (A. Pura-pura mendengarkan ceritanya dan fokus pada pekerjaan saya)

-Score 1 :mungkin menunjukkan kurangnya kesadaran atau perhatian terhadap prinsip-prinsip moral yang mendasari situasi yang dihadapi. Jawaban ini mungkin lebih cenderung untuk mengabaikan implikasi etis dari tindakan yang diambil, dengan lebih memperhatikan kepentingan pribadi atau keuntungan langsung tanpa mempertimbangkan konsekuensi moral yang lebih luas (D. Menanggapi dan memberi berbagai alternatif solusi)


{explanation}
(Dari struktur answers[option, answer, order, score, is_true], berikan penjelasan untuk setiap answer. Selalu gunakan answer dari answers, jangan menggunakan indikator opsinya untuk merujuk pada answer yang dimaksud. Jelaskan secara detail pada tiap answer mengapa answer tersebut benar atau mengapa answer tersebut kurang benar, gunakan format sebagai berikut:

Jawaban Yang Benar: [Tulis isi dari answer tanpa indikator opsi] dengan skor (skornya), lalu pembahasan mengapa answer tersebut benar.

Jawaban yang salah: [Berisikan pembahasan masing-masing answer lainnya (hanya tulis isi dari answer tanpa indikator opsi)] dengan skor (skornya), dan mengapa answer tersebut mendapat skor yang demikian.

Jangan gunakan formatting lain selain HTML)

Pilihan ganda dibuat sekreatif mungkin dengan 5 opsi. Opsi jawaban harus beragam dan logis, namun gunakan pengecoh yang mirip untuk menyamarkan kunci jawaban. Soal harus memenuhi kaidah penulisan soal pilihan ganda yang baik dan benar.

Berikut adalah contoh respons yang benar bahkan dengan opsi yang mirip sehingga menyusahkan peserta untuk menjawab soal:

{format}

SELALU GUNAKAN TAG HTML UNTUK FORMATTING

dari instruksi tersebut lakukan task berikut

Task: 

"{task}"

Walupun Task yang diberikan terlalu rumit sehingga kamu membuat soal yang singkat namun coba cari tahu terlebih dahulu maksud dari topik yang diminta utamakan hasilkan soal yang kreatif dan konteks yang panjang jangan yang singkat. Walaupun soal perlu melenceng sedikit dari topik sedikit tidak masalah yang penting hasil soalnya panjang ceritanya bisa dibuat sebebas mungkin.
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
        description="berisi soal HIGH ORDER THINKING SKILLS yang harus sesuai kriteria yang diminta, pastikan soal memiliki konteks yang panjang untuk menambah kompleksitas soal. minimal 780 character. question tidak boleh terlalu singkat atau pendek semakin panjang konteks soal semakin baik")
    answers: list = Field(
        description="list yang berisikan option jawaban yang harus sesuai kriteria yang diminta. Yaitu setiap opsi minimal memiliki 15 words")
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

