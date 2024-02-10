from DataPreparation import DataPreparation
from RecomendProcess import ModelsRunner


def main(mConfig: dict,
         resume: str,
         regrConfig: dict,
         dataPath: str = './data/database.csv',
         pathLemmasTexts: str = './data/prepdf9000.csv',
         nameClustModel: str = 'LdaModel9000.pkl',
         regrNameModel: str = './models/CatBoostModel10500.cbm',
         Nrecs: int = 10,
         NrecSkills: int = 10,
         modelType: str = 'LDA',
         oneHotSkillsPath: str = './data/oneHotSkills9000.csv'):

    dataPrep = DataPreparation(dataPath)
    dataPrep.run(baseTokenIsSkills=True,
                 pathSaveLemmasTexts=pathLemmasTexts,
                 oneHotSavePath=oneHotSkillsPath)

    mr = ModelsRunner(prepareDF=dataPrep.prepDF,
                      descriptionRegexPattern=dataPrep.regexPatterns['Description'],
                      vocab=dataPrep.skillSet,
                      oneHotSkills=oneHotSkillsPath,
                      modelType=modelType,
                      modelPath=nameClustModel,
                      modelConfig=mConfig,
                      regrConfig=regrConfig,
                      regrModelPath=regrNameModel)

    cluster_num, prep_resume = mr.run_process(resume=resume)
    mr.run_recomends(clust=cluster_num,
                     prepResume=prep_resume,
                     nRecVacs=Nrecs,
                     nRecSkills=NrecSkills,
                     pathOrigData=dataPath)


if __name__ == '__main__':
    dataPath: str = './data/db10500.csv'
    pathLemmasTexts: str = './data/prepdf10500.csv'
    oneHotSkill: str = './data/OHS10500.csv'
    NVacRecs: int = 5
    NskillsRecs: int = 5
    regrModelName = './models/CatBoostModel10500.cbm'
    resume = 'design, photoshop, figma, ux, ui, 3d'
    topicModelType = 'NMF'


    regrConfig = {'text_features': ['Description'],
                  'cat_features': ['Experience', 'Schedule'],
                  'loss_function': "RMSE",
                  'learning_rate': 0.25,
                  'iterations': 300,
                  'depth': 7,
                  'verbose': False,
                  'random_state': 0,
                  'task_type': "GPU"}

    if topicModelType == 'LDA':
        modelConfig = {"num_topics": 70,
                       "random_state": 0,
                       "update_every": 1,
                       "chunksize": 100,
                       'minimum_probability': 0.005}
        modelName = './models/LdaModel10500.pkl'

    elif topicModelType == 'NMF':
        modelConfig = {'n_components': 120,
                       'random_state': 0,
                       'solver': 'mu',
                       'beta_loss': 'kullback-leibler'}
        modelName = './models/NMFmodel10500.pkl'

    main(regrConfig=regrConfig,
         dataPath=dataPath,
         pathLemmasTexts=pathLemmasTexts,
         Nrecs=NVacRecs,
         NrecSkills=NskillsRecs,
         mConfig=modelConfig,
         nameClustModel=modelName,
         resume=resume,
         modelType=topicModelType,
         oneHotSkillsPath=oneHotSkill)