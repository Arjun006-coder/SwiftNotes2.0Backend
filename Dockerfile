# Use a lightweight official Python 3.12 image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirement list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend files
COPY . .

# Expose the correct port for Google Cloud Run (defaults to 8080)
ENV PORT=8080
EXPOSE 8080

# Command to run the uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
