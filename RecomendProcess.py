import re
import os
import pandas as pd
import numpy as np
from utils import lemmatize, saveData, loadData
from modelBuilder import LDAmodel, NMFmodel, CatBoostModel
from collections import Counter


class ModelsRunner:
    def __init__(self,
                 prepareDF: pd.DataFrame,
                 descriptionRegexPattern: str,
                 vocab: list,
                 oneHotSkills: str,
                 modelPath: str,
                 modelType: str,
                 modelConfig: dict,
                 regrConfig: dict,
                 regrModelPath: str):
        self.resDF = prepareDF
        self.id2word = None
        self.encodeCorpus = None
        self.descRP = descriptionRegexPattern

        self.vocab = vocab
        self.oneHotSkills = oneHotSkills

        self.modelWrap = None
        self.modelType = modelType
        self.modelPath = modelPath
        self.modelConfig = modelConfig

        self.regrConfig = regrConfig
        self.regrPath = regrModelPath

    def run_process(self, resume: str):
        # init model object
        self.oneHotSkills = loadData(self.oneHotSkills)

        if self.modelType == 'LDA':
            modelWrap = LDAmodel(self.resDF, descriptionRegexPattern=self.descRP,
                                 vocab=self.vocab, oneHotSkills=self.oneHotSkills)
        else:
            modelWrap = NMFmodel(self.resDF, descriptionRegexPattern=self.descRP,
                                 vocab=self.vocab, oneHotSkills=self.oneHotSkills)

        # fit model
        modelWrap.prepare_input()
        if not os.path.exists(self.modelPath):
            modelWrap.fit(modelConfig=self.modelConfig, savePath=self.modelPath)
        else:
            modelWrap.model = loadData(self.modelPath)

        # eval model
        modelWrap.predict()
        modelWrap.model_eval(topicTermData='./results/descriptionTopics.csv')
        clust, prepResume = modelWrap.inference(resume)

        self.modelWrap = modelWrap
        return clust, prepResume

    def recomend_prof(self, listProffesions: list[str], stopwordsProfs: list[str]) -> None | dict | str :
        normProfName = []
        for prof in listProffesions:
            normProfName.append(' '.join(set(lemmatize(prof, self.descRP,
                                          stops=stopwordsProfs,
                                          tokens=self.vocab).split())))

        for x in Counter(normProfName).most_common():
            if len(x[0].split()) > 1:
                print("\n- Рекомендуемая профессия: " + x[0])
                return "Рекомендуемая профессия: " + x[0]
                break

    def recomend_skills(self, clust: int,
                        prepResume: list[str],
                        nRecSkills: int) -> None | str:
        top_terms = []
        if self.modelType == "LDA":
            top_terms = self.modelWrap.model.print_topic(clust, topn=len(prepResume) + int(nRecSkills * 1.5))
            # out in top_terms for example: 0.042*"react" + 0.040*"js" + 0.030*"git" + 0.029*"frontend"
            top_terms = re.findall(re.compile(r'"\w+"'), top_terms)
            top_terms = [x[1:-1] for x in top_terms if '_' not in x]

        elif self.modelType == "NMF":
            feature_names = self.modelWrap.vectorizer.get_feature_names_out()
            top_ids_words = self.modelWrap.model.components_[clust].argsort()[-(len(prepResume) + int(nRecSkills * 1.5)):][
                            ::-1]
            top_terms = [feature_names[i] for i in top_ids_words]

        outerSkills = list(set(top_terms) - set(prepResume))
        print("\n- Самые распространненые навыки в кластере профессий:")
        print(outerSkills)

        simCosine = np.zeros(shape=(len(prepResume), len(outerSkills)))
        for i, resumeSkill in enumerate(prepResume):
            a = self.oneHotSkills.loc[:, resumeSkill].values
            for j, clustSkill in enumerate(outerSkills):
                b = self.oneHotSkills.loc[:, clustSkill].values
                simCosine[i, j] = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

        recommend_skills = " Рекомедуем Вам изучить следующие навыки, в котексте следущих компаний: "
        print("\n- Рекомедуем Вам изучить следующие навыки(частота встречаемости с вашими навыками)")
        topInds = np.argpartition(simCosine.mean(axis=0), kth=-nRecSkills)[-nRecSkills:]
        for i, x in enumerate(topInds):
            recommend_skills += f'{i+1}) ' + outerSkills[x] + '  '
            print(f'%-20s| %1.5f' % (f'{i+1}) ' + outerSkills[x], simCosine.mean(axis=0)[x]), end='\n')
        print('\n')
        return recommend_skills.rstrip(" ")

    def recomend_vacancies(self,
                           clust: int,
                           prepResume: list[str],
                           nRecVacs: int,
                           pathOrigData: str,
                           pathSaveResultVacs: str) -> None | pd.DataFrame:
        print('Подбор подходящих вакансий...', end=' ')
        resumeTokenVect = np.array([1 if token in prepResume else 0 for token in self.vocab], dtype=np.uint)
        currentClustVacs = self.oneHotSkills[(self.modelWrap.resDF['TopicLabel'] == clust).values]
        cosMetr = currentClustVacs.values.dot(resumeTokenVect) / np.linalg.norm(currentClustVacs.values, axis=1)
        topCos = np.argpartition(cosMetr, kth=-nRecVacs)[-nRecVacs:]
        topVacsIndex = currentClustVacs.index[topCos]

        recVacsDF = self.modelWrap.resDF.iloc[topVacsIndex, :]
        dataOrig = pd.read_csv(pathOrigData, index_col=0)

        useColumns = ['Ids', 'Employer', 'Name', 'Salary', 'From', 'To', 'Experience', 'Schedule', 'Keys',
                      'Description']
        drop_columns = set(dataOrig.columns) - set(useColumns)
        dataOrig.drop(columns=drop_columns, axis=1, inplace=True)
        recDf = dataOrig.iloc[recVacsDF.index, :]
        recDf['resume similarity'] = cosMetr[topCos]
        print('Вакансии подобраны.', end=' ')
        saveData(recDf, pathSaveResultVacs)
        return recDf

    def recomend_salary(self, prepResume: list[str]):
        print('Оценка средней зароботной платы...', end=' ')
        salaryModel = CatBoostModel(config=self.regrConfig)
        target = self.modelWrap.resDF[self.modelWrap.resDF['Salary'] & ~self.modelWrap.resDF['Description'].isnull()][
            ['From', 'To']].mean(axis=1)
        target = target[(target < 500000) & (target > 40000)]
        X_data = self.modelWrap.resDF.loc[target.index, ['Schedule', 'Experience', 'Description']]
        salaryModel.train(X_data, target.values, self.regrPath)

        print('Оценка завершена.', end=' ')
        saveData(salaryModel.inference(' '.join(prepResume)),
                 './results/SalaryEstimation.csv')

    def run_recomends(self,
                      clust: int,
                      prepResume: list,
                      nRecVacs: int = 5,
                      nRecSkills: int = 5,
                      pathOrigData:str = './data/database.csv',
                      pathSaveRecsVacs: str = './results/Recomendations.csv') -> None | dict:

        nameProfs = self.modelWrap.resDF[self.modelWrap.resDF['TopicLabel'] == clust]['Name'].values

        recommend_prof = self.recomend_prof(listProffesions=nameProfs, stopwordsProfs=['junior', 'senior', 'middle',
                                                                      'старший', 'младший'])

        recommend_skills = self.recomend_skills(clust=clust,
                             prepResume=prepResume,
                             nRecSkills=nRecSkills)

        recommend_vacs = self.recomend_vacancies(clust=clust,
                                prepResume=prepResume,
                                nRecVacs=nRecVacs,
                                pathOrigData=pathOrigData,
                                pathSaveResultVacs=pathSaveRecsVacs)

        self.recomend_salary(prepResume=prepResume)

        return {'professions': recommend_prof,
                'skills': recommend_skills,
                'vacancies_df': recommend_vacs}






