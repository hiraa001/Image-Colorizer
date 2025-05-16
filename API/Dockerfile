# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install git to clone the repository
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Clone the Image-Colorizer repository
RUN git clone https://github.com/fatimaazfar/Image-Colorizer.git

# Set the working directory to the API folder
WORKDIR /app/Image-Colorizer/API

# Install the required Python packages
RUN pip install --no-cache-dir fastapi uvicorn torch numpy Pillow

# Expose the port that the FastAPI app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]