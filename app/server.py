from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
import sys
sys.path.append("/Users/restuprajna/Developer/langchain-app/vertex-ai/my-app/packages/openai-api")

# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")

@app.get("/")
def root():
    return {
        "message": "Welcome to the langserve server! More information and routes can be found at /docs. 🦜 🏓",
    }

import sys
print(sys.path)

# Edit this to add the chain you want to add
from rag_google_cloud_vertexai_search import chain as rag_google_cloud_vertexai_search_chain

add_routes(app, rag_google_cloud_vertexai_search_chain, path="/rag-google-cloud-vertexai-search")

from openai_api import chain as openai_api_chain

add_routes(app, openai_api_chain, path="/openai-api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
