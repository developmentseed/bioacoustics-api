FROM python:3.8-slim

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./* .

# to make sure logs are printed instantly without buffering
ENV PYTHONUNBUFFERED=1

CMD ["python", "./run.py", "--overwrite-collection", "--load-percentage", "25", "--steps", "all"]
