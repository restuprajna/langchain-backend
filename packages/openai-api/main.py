from openai_api.chain import chain

if __name__ == "__main__":
    query = "Who is the CEO of Google Cloud?"
    print(chain.invoke(query))
