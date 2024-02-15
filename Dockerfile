FROM python:3.11-slim

RUN pip install --upgrade pip
RUN pip install langchain
RUN pip install -U langchain-cli
RUN pip install langchain_google_genai
RUN pip install -U langchain-community
ENV GIT_PYTHON_REFRESH=quiet

ENV PORT=8080

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /app

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./package[s] ./packages

RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

RUN poetry install --no-interaction --no-ansi

EXPOSE $PORT


# COPY requirements.txt /my-app/requirements.txt
# RUN pip install -r requirements.txt

# RUN pip install python-dotenv

EXPOSE 8080

# CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080

CMD ["langchain", "serve", "--host", "0.0.0.0", "--port", "$PORT"]
