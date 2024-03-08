from fastapi import FastAPI
from fastapi import Depends, Request, APIRouter
from fastapi.responses import RedirectResponse
from langserve import add_routes
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, status, Depends, Request, Response
from fastapi.security import APIKeyHeader, HTTPBearer
from langserve import APIHandler
# from langsmith import Client

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
sys.path.append("../packages")


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
from openai_api import chain as openai_api_chain

# add_routes(app, rag_google_cloud_vertexai_search_chain, path="/rag-google-cloud-vertexai-search")
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
