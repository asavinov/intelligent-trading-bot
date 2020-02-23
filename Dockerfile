FROM python:3.7-slim

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  gcc \
  python3.7-dev \
  && rm -rf /var/lib/apt/lists/*

# MAINTAINER Alexandr Savinov "asavinov@yahoo.com"

# Install package
COPY ./requirements.txt /requirements.txt
#ADD requirements.txt /app
RUN pip install -r requirements.txt

# Copy application
#RUN mkdir /app
COPY . /app

WORKDIR /app/trade

#ENV SITE_URL http://example.com/
#WORKDIR /data
#VOLUME /data

RUN pwd
RUN ls -a

# Start application
CMD [ "python", "main.py" ]
