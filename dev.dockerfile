FROM python:3.13.1

WORKDIR /app

COPY /requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

# Expose FastAPI port
EXPOSE 8000

# Default command: run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
