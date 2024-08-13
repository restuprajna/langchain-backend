from fastapi import FastAPI
from fastapi import Depends, Request, APIRouter
from fastapi.responses import RedirectResponse
from langserve import add_routes
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, status, Depends, Request, Response, File, UploadFile, Body
from fastapi.responses import JSONResponse
import shutil
from fastapi.security import APIKeyHeader, HTTPBearer
from langserve import APIHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import base64
# from langsmith import Client
from bson import ObjectId

import os
load_dotenv()

app = FastAPI()

# client = Client()

# add comment teast dev

# Auth bearer Method
bearer = HTTPBearer()

async def validate_token(token: str = Depends(bearer)):
    # Retrieve the expected API key from the environment variable
    expected_api_key = os.getenv("THIS_IS_AI_API_TOKEN")
    # print(token)
    # print(expected_api_key)
    # Validate the Bearer Token
    if token.credentials != expected_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token

# Asynchronous function to retrieve the expected API key from the environment variable
# async def get_expected_api_key():
#     return os.getenv("OPENAI_API_KEY")

import sys
# sys.path.append("../packages")
sys.path.append(os.path.abspath("../packages"))
print(sys.path)

# packages/rag-google-cloud-vertexai-search/rag_google_cloud_vertexai_search


# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")

@app.get("/")
def root():
    return {
        "message": "Welcome to the langserve server! More information and routes can be found at /docs. 🦜 🏓",
    }


@app.get("/protected")
async def protected_route(token: str = Depends(validate_token)):
    return {"message": "You are authorized!"}


# Edit this to add the chain you want to add
from rag_google_cloud_vertexai_search import chain as rag_google_cloud_vertexai_search_chain
# from openai_api import chain as openai_api_chain
# from vertex_tkp_chain import chain as vertex_tkp_chain
# from mongo_rag import chain as mongo_rag_chain
# from soal_pppk import chain as soal_pppk_chain
# from generate_goals import chain as generate_goals_chain

add_routes(app, mongo_rag_chain, path="/test-api")
vertex_api_handler = APIHandler(rag_google_cloud_vertexai_search_chain, path="/vertex-ai")
@app.post("/vertex-ai/{instance_id}/invoke", include_in_schema=True)
async def protected_route_openai(instance_id: str, token: str = Depends(validate_token), request: Request = None):
    """
    Route protected by token validation.
    """
    path = f"/vertex-ai/{instance_id}/invoke"
    response = await invoke_api(rag_google_cloud_vertexai_search_chain, path, request)
    return response

@app.post("/vertex-ai/{instance_id}/batch", include_in_schema=True)
async def protected_route_openai(instance_id: str, token: str = Depends(validate_token), request: Request = None):
    """
    Route protected by token validation.
    """
    path = f"/vertex-ai/{instance_id}/batch"
    response = await batch_api(rag_google_cloud_vertexai_search_chain, path, request)
    return response

@app.post("/vertex-ai/{instance_id}/soal-tkp/batch", include_in_schema=True)
async def protected_route_openai(instance_id: str, token: str = Depends(validate_token), request: Request = None):
    """
    Route protected by token validation.
    """
    path = f"/vertex-ai/{instance_id}/soal-tkp/batch"
    response = await batch_api(vertex_tkp_chain, path, request)
    return response

@app.post("/vertex-ai/{instance_id}/soal-pppk/batch", include_in_schema=True)
async def protected_route_openai(instance_id: str, token: str = Depends(validate_token), request: Request = None):
    """
    Route protected by token validation.
    """
    path = f"/vertex-ai/{instance_id}/soal-pppk/batch"
    response = await batch_api(soal_pppk_chain, path, request)
    return response

@app.post("/vertex-ai/{instance_id}/soal-pppk/invoke", include_in_schema=True)
async def protected_route_openai(instance_id: str, token: str = Depends(validate_token), request: Request = None):
    """
    Route protected by token validation.
    """
    path = f"/vertex-ai/{instance_id}/soal-pppk/invoke"
    response = await invoke_api(soal_pppk_chain, path, request)
    return response


# @app.post("/vertex-ai/{instance_id}/mongo-rag/batch", include_in_schema=True)
# async def protected_route_openai(instance_id: str, token: str = Depends(validate_token), request: Request = None):
#     """
#     Route protected by token validation.
#     """
#     path = f"/vertex-ai/{instance_id}/mongo-rag/batch"
#     response = await batch_api(mongo_rag_chain, path, request)
#     return response

@app.post("/vertex-ai/{instance_id}/mongo-rag/batch", include_in_schema=True)
async def protected_route_openai(instance_id: str, token: str = Depends(validate_token), request: Request = None):
    """
    Route protected by token validation.
    """
    path = f"/vertex-ai/{instance_id}/mongo-rag/batch"
    response = await batch_api(mongo_rag_chain, path, request)
    
    # Assuming response is a dictionary containing the parsed JSON response
    # category_value = response.get("kwargs", {})
    
    return response
    # return {
    #     "response": response,
    #     # "category" : instance_id
    # }


@app.post("/vertex-ai/{instance_id}/genereate-goals/batch", include_in_schema=True)
async def protected_route_openai(instance_id: str, token: str = Depends(validate_token), request: Request = None):
    """
    Route protected by token validation.
    """
    path = f"/vertex-ai/{instance_id}/generate-goals/batch"
    response = await batch_api(generate_goals_chain, path, request)
    return response






# add_routes(app, openai_api_chain, path="/openai-api")

@app.post("/openai-api/{instance_id}/invoke", include_in_schema=True)
async def protected_route_openai(instance_id: str, token: str = Depends(validate_token), request: Request = None):
    """
    Route protected by token validation.
    """
    path = f"/openai-api/{instance_id}/invoke"
    response = await invoke_api(openai_api_chain, path, request)
    return response

@app.post("/openai-api/{instance_id}/batch", include_in_schema=True)
async def protected_route_openai(instance_id: str, token: str = Depends(validate_token), request: Request = None):
    """
    Route protected by token validation.
    """
    path = f"/openai-api/{instance_id}/batch"
    response = await batch_api(openai_api_chain, path, request)
    return response


# define invoke, batch and stream method

async def invoke_api(api_chain, path: str, request: Request) -> Response:
    """
    Handle a request for the specified API.
    """
    api_handler = APIHandler(api_chain, path=path)
    return await api_handler.invoke(request)



async def batch_api(api_chain, path: str, request: Request) -> Response:
    """
    Handle a request for the specified API.
    """
    api_handler = APIHandler(api_chain, path=path)
    return await api_handler.batch(request)



# UPLOAD FILE TO VECTOR DB

@app.post("/upload_file", include_in_schema=True)
async def upload_file(uploaded_file: UploadFile = File(...), category_id: str = Body(...)):
    try:
        # Process the uploaded file
        # file_content = await file.read()  # Read file content as bytes
        # with open(file.filename, 'wb') as f:
        #     f.write(file_content)

        file_location = uploaded_file.filename
        with open(file_location, "wb+") as file_object:
            file_object.write(uploaded_file.file.read())

        return process_file(file_location, category_id)
    except Exception as e:
        # Handle any exceptions that occur during file processing
        raise HTTPException(status_code=500, detail=str(e))

google_api: str = os.environ["GOOGLE_API_KEY"]
atlas_cluster_uri: str = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
client = MongoClient(atlas_cluster_uri)

DB_NAME: str = os.environ["DB_NAME"]
EMBEDDING_COLLECTION_NAME: str = os.environ["EMBEDDING_COLLECTION"]
ATLAS_VECTOR_SEARCH_INDEX_NAME: str = os.environ["ATLAS_VECTOR_SEARCH_INDEX_NAME"]

DOCUMENT_COLLECTION = client[DB_NAME][EMBEDDING_COLLECTION_NAME]
embedding_model:str = os.environ["embedding_model"]

# def add_category(category: str):
#     global DB_NAME
#     global client

#     CATEGORY_COLLECTION_NAME: str = os.environ["CATEGORY_COLLECTION"]

#     CATEGORY_COLLECTION = client[DB_NAME][CATEGORY_COLLECTION_NAME]

#     category_doc = {
#         "category": category
#     }
#     category_res = CATEGORY_COLLECTION.insert_one(category_doc)
#     category_id = category_res.inserted_id
#     return category_id

def process_file(file_content: str, category_id: str):
    global google_api
    global atlas_cluster_uri
    global DB_NAME
    global EMBEDDING_COLLECTION_NAME
    global ATLAS_VECTOR_SEARCH_INDEX_NAME
    global DOCUMENT_COLLECTION
    global embedding_model

    print(type(file_content))
    # initialize MongoDB python client

    try:
        # Load the PDF file from the uploaded file content
        

        loader = PyPDFLoader(file_content)
        pages = loader.load()

        # Split the pages into chunks
        pages = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000))

        # category_id = add_category(category)
        # Add the category metadata to each chunk
        for page in pages:
            page.metadata['category_id'] = ObjectId(category_id)

        # Declare the embedding model
        google_embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=google_api)

        # Declare the vector store
        vector_store = MongoDBAtlasVectorSearch(
            embedding=google_embeddings,
            collection=DOCUMENT_COLLECTION,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
        )

        # Insert the document chunks into the vector store
        vector_search_ingest = MongoDBAtlasVectorSearch.from_documents(
            documents=pages,
            embedding=google_embeddings,
            collection=DOCUMENT_COLLECTION,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
        )

        # Return a success message
        os.remove(file_content)
        return {"message": "File processed and uploaded to vector database."}
    except Exception as e:
        # Handle any exceptions that occur during file processing
        os.remove(file_content)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
