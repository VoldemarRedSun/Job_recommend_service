FROM python:3.10

WORKDIR /app


COPY . .


RUN ./scripts/data_download.sh  && ./scripts/models_download.sh

RUN pip install -r requirements.txt


 # Run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]