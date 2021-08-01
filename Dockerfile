#Base Image to use
#From pytorch/pytorch-binary-docker-image-ubuntu16.04
FROM python:3.8-buster

#Expose port 8080
EXPOSE 8080

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "webapp.py", "--server.port=8080", "--server.address=0.0.0.0"]
