# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

RUN apt-get -y update  \
    && apt-get install -y ffmpeg  \
    && apt-get install -y libpq-dev  \
    && apt-get install -y build-essential

# Install the dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

RUN apt-get update && apt-get install -y curl

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
