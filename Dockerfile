FROM python:3.11-slim

RUN pip install --upgrade pip
# RUN pip install langchain
# RUN pip install -U langchain-cli
# RUN pip install langchain_google_genai
# RUN pip install -U langchain-community

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENV GIT_PYTHON_REFRESH=quiet


RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /app

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./packages ./packages

RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

RUN poetry install --no-interaction --no-ansi



EXPOSE 8080

ENV PORT=8080



# COPY requirements.txt /my-app/requirements.txt
# RUN pip install -r requirements.txt

# RUN pip install python-dotenv

# EXPOSE 8080

CMD exec uvicorn app.server:app --host 0.0.0.0 --port $PORT

# command in hot reload
# CMD exec uvicorn app.server:app --host 0.0.0.0 --port $PORT --reload

# docker build image 
# docker build -t langchain-api

# docker start container
# docker run -p 80:8080 --env-file .env --name langchain-api -v ./:/app -d --restart unless-stopped langchain-api


# docker restart container
# docker container restart <container-name>
