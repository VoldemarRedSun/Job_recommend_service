FROM python:3.10

WORKDIR /app

COPY . .

RUN pwd

RUN ls -l

RUN ./scripts/data_download.sh

RUN ls -l


CMD [ "python", "./my_test.py" ]