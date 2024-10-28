FROM python:3.11-slim

# Add user and change working directory
RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /home/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create saves directory and set permissions
RUN mkdir -p saves && \
    chown -R app:app /home/app

# Switch to non-root user
USER app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
ENTRYPOINT ["shiny", "run", "--host", "0.0.0.0", "--port", "8000"]
