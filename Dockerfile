#Deploy from GCP
FROM python:3.8-buster

#Expose port 8051
EXPOSE 8051

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8051
ENTRYPOINT ["streamlit", "run", "fostercarematcher.py", "--server.port=8051", "--server.address=0.0.0.0"]

