FROM python:3.13

WORKDIR /cardiocare

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY data/ .
COPY src/ .
COPY tests/ .

ENTRYPOINT ["python", "inference.py"]