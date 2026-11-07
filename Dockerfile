FROM python:3.13-slim

WORKDIR /app

# CPU-only torch wheels: containers on macOS cannot use MPS, and the default
# x86 wheels would pull several GB of CUDA libraries.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY *.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
