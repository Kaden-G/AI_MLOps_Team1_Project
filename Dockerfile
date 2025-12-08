FROM python:3.12-slim

RUN pip install uv

COPY . /code/

WORKDIR /code

RUN uv venv --python 3.12

RUN uv pip install -r pyproject.toml

ENV PATH="/code/.venv/bin:$PATH"

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]
