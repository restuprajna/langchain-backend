from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")

@app.get("/")
def root():
    return {
        "message": "Welcome to the langserve server! More information and routes can be found at /docs. 🦜 🏓",
    }


# Edit this to add the chain you want to add
from rag_google_cloud_vertexai_search import chain as rag_google_cloud_vertexai_search_chain

add_routes(app, rag_google_cloud_vertexai_search_chain, path="/rag-google-cloud-vertexai-search")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
