# ML job recommendation service on FastApi

This service recommends according to your competencies the most suitable profession for you, the skills that are relevant to it and a list of suitable vacancies (the database of vacancies was collected by parsing with hh.ru )

## Local development

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv/Source/activate

# Install dependencies
pip install -r requirements.txt

# Download data and modles
scripts/data_download.sh
scripts/models_download.sh

# Run app
uvicorn app.app:app --host 0.0.0.0 --port 8080

# Use swagge APi to access the app
http://0.0.0.0:8080/docs
or 
http://localhost:8080/docs
```

## Run app in docker container

```bash
docker build -t ml-app .
docker run -p 80:80 ml-app
```
