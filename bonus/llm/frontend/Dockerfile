FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV BACKEND_URL=http://backend:8002

EXPOSE 7862

CMD ["python", "app.py"]
