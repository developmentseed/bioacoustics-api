FROM python:3.8-slim

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./* .

CMD ["bash", "./run.sh", "--overwrite-collection"]
