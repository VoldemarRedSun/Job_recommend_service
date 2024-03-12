from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from main import main



tags_metadata = [
    {
        "name": "predict",
        "description": "predicts the most suitable profession, required skills and vacancies with hh.ru "
    },

]

app = FastAPI(openapi_tags=tags_metadata)



class VacancyResponse(BaseModel):
    professions: str
    skills: str
    vacancies: dict




@app.get("/predict", tags = ["predict"] )
def predict(resume: str, topicModelType: str):
    dataPath: str = './data/db10500.csv'
    pathLemmasTexts: str = './data/prepdf10500.csv'
    oneHotSkill: str = './data/OHS10500.csv'
    NVacRecs: int = 5
    NskillsRecs: int = 5
    regrModelName = './models/CatBoostModel10500.cbm'



    regrConfig = {'text_features': ['Description'],
                  'cat_features': ['Experience', 'Schedule'],
                  'loss_function': "RMSE",
                  'learning_rate': 0.25,
                  'iterations': 300,
                  'depth': 7,
                  'verbose': False,
                  'random_state': 0,
                  'task_type': "GPU"}


    topicModelType = NMF
    modelConfig = {'n_components': 120,
                       'random_state': 0,
                       'solver': 'mu',
                       'beta_loss': 'kullback-leibler'}
    modelName = './models/NMFmodel10500.pkl'


    prediction = main(regrConfig=regrConfig,
                 dataPath=dataPath,
                 pathLemmasTexts=pathLemmasTexts,
                 Nrecs=NVacRecs,
                 NrecSkills=NskillsRecs,
                 mConfig=modelConfig,
                 nameClustModel=modelName,
                 resume=resume,
                 modelType=topicModelType,
                 oneHotSkillsPath=oneHotSkill)
    response = VacancyResponse(
        professions = prediction['professions'],
        skills = prediction['skills'],
        vacancies = prediction['vacancies_df'].to_dict()['Name']
    )

    return response