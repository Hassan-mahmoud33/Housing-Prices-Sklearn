# Base image with Python already installed
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app
 
# Copy requirements first (this layer gets cached, so rebuilds are faster
# when only your code changes and not your dependencies)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# Now copy the rest of the project (api/, front_end/, models/)
COPY . . 

# The port uvicorn will listen on inside the container
EXPOSE  8000

# Start the API. api.main:app means: file api/main.py, variable app
CMD ["uvicorn" , "api.main:app" , "--host" , "0.0.0.0" , "--port" , "8000"]