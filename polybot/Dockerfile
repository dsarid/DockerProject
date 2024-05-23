FROM python:3.10.14-alpine

# Set the working directory
WORKDIR /usr/src/app

# Copy requirements.txt and install dependencies
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install --upgrade pip
RUN pip cache purge
RUN pip install --no-cache-dir matplotlib
# Copy the rest of the application files
COPY . .

# Specify the command to run the application
CMD ["python3", "app.py"]
