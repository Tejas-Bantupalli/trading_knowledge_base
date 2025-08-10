FROM python:3.9-alpine

# Install system dependencies
RUN apk add --no-cache gcc musl-dev libffi-dev postgresql-dev

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Command to run the application
CMD ["python", "app.py"]
