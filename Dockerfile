FROM python:3.9
ENV PYTHONUNBUFFERED 1

RUN apt-get -qq -y update \
    && apt-get install -y libgeos-dev python3-gdal postgresql-client libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Allows docker to cache installed dependencies between builds
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Adds our application code to the image
COPY . code
WORKDIR code

EXPOSE 8000

# Run the production server
CMD gunicorn --bind 0.0.0.0:8000 --access-logfile - bioacoustics.wsgi:application
