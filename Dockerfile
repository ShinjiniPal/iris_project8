# Use an official Python runtime as a parent image
FROM python:3.9.7-slim

# Set the working directory to /app
WORKDIR /app

# Install build dependencies
RUN apt-get update \
    && apt-get install -y gcc libffi-dev libssl-dev

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Expose port 5000 for the Flask application
EXPOSE 5000

# Use Gunicorn to run the Flask application
ENTRYPOINT ["gunicorn", "second:app", "-b", "0.0.0.0:5000"]
