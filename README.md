# Backend for Lanchain App

this app is an example of prompt engineering with steroid. This app implement a technique call RAG (Retrieval Augmented Generation) to adding private knowledge to LLM models without the use fo Fine tune. 

![Uploading image.png…]()


## Installation

Install the necessary package

```bash
pip install -r requirements.txt
```

## Adding packages

```bash
# adding packages from 
# https://github.com/langchain-ai/langchain/tree/master/templates
langchain app add $PROJECT_NAME

# adding custom GitHub repo packages
langchain app add --repo $OWNER/$REPO
# or with whole git string (supports other git providers):
# langchain app add git+https://github.com/hwchase17/chain-of-verification

# with a custom api mount point (defaults to `/{package_name}`)
langchain app add $PROJECT_NAME --api_path=/my/custom/path/rag
```

Note: you remove packages by their api path

```bash
langchain app remove my/custom/path/rag
```


## Launch LangServe

```bash
langchain serve
```


Adiing local package to server.py
```bash
pip install -e ./packages/openai-api
```



