FROM python:3.10-slim

# MANDATORY: Install the C++ library that LightGBM needs to run
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for better build speed)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

EXPOSE 8000

# Use python -m to ensure the paths are handled correctly
CMD ["python", "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]