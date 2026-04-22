FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Ports: 8000 = FastAPI, 8501 = Streamlit, 5000 = MLflow UI
EXPOSE 8000 8501 5000
