FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY implementation/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy application code
COPY implementation/ implementation/
COPY geokniga-drillingengineeringprasslwl.pdf .

# Expose port
EXPOSE 7860

# Run the app (HF Spaces expects port 7860)
CMD ["uvicorn", "implementation.app:app", "--host", "0.0.0.0", "--port", "7860"]
