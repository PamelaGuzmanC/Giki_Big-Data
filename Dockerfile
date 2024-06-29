# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt into the container at /app
COPY app/requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY app/ /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]
