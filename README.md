# Backend for Lanchain App

this app is an example of prompt engineering with steroid. Leveraging a technique called Retrieval Augmented Generation (RAG). RAG enhances large language models by integrating private knowledge without the need for fine-tuning.

![Alt text](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*J7vyY3EjY46AlduMvr9FbQ.png)


Here's how RAG works:

1. Create Chunks of Text: Start by segmenting your desired knowledge, such as private company data (e.g., a list of employees), into manageable chunks.

2. Vectorization: Convert these text chunks into vectors (numerical representations) and store them in a vector database. In this app, MongoDB Atlas is used for this purpose.

3. Semantic Search: When you ask a question involving private knowledge, such as "Who is the head of marketing at Company X?", the system performs a semantic search in the vector database to find the most relevant information.

4. Augment the Prompt: The retrieved information is then added to the prompt, which is sent to the large language model (LLM).

5. Get Your Answer: The LLM processes the augmented prompt and provides an accurate answer, seamlessly integrating the private knowledge.

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



