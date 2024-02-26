import re
import pymorphy2
import os
import pandas as pd
import pickle

lemmatizer = pymorphy2.MorphAnalyzer(lang='ru')

REPLACE_TOKENS = {'k8s': 'kubernetes',
                  'джава': 'java',
                  'javascript': 'js',
                  'с++': 'c++', # first symbol is russian, second is english
                  ' си ': ' c ',
                  '1c': '1с',
                  '++': '` `',
                  '+': ' ',
                  '` `': '++',
                  'питон': 'python',
                  'пайтон': 'python',
                  'tf': 'tensorflow'}

def token_replace(string):
    rep = dict((re.escape(k), v) for k, v in REPLACE_TOKENS.items())
    pattern = re.compile("|".join(rep.keys()))
    string = pattern.sub(lambda m: rep[re.escape(m.group(0))], string)
    return string

def strDictParse(x: str, pattern: str,
                 integr: bool = False,
                 leftBound: int = 0,
                 rightBound: int = 0,
                 save_all: bool = False) -> list:
    if x != x:
        return None
    else:
        s = re.findall(pattern, x)
        if len(s) < 1:
            return None
        else:
            if not save_all:
                s = s[0]
                if rightBound == 0: rightBound = len(s)
                return int(s[leftBound:rightBound]) if integr else s[leftBound:rightBound]
            else:
                return [skill[leftBound:rightBound].lower() for skill in s]


def lemmatize(text: str, delSymbPattern: str,
              stops: list = [],
              tokens: list = None,
              bounds: bool = True,
              centerSlice: float = 0.5,
              sliceRadius: int = 100) -> str:

    text_preps = re.sub(delSymbPattern, ' ', text.lower())
    text_preps = token_replace(text_preps)

    lenText = len(text_preps.split())
    if tokens:
        s = []
        for word in text_preps.split():
            prep_word = lemmatizer.parse(word)[0].normal_form
            if (prep_word in tokens) and (prep_word not in stops):
                s.append(prep_word)
        out = ' '.join(s)

    else:
        if bounds and (lenText > sliceRadius/(1-centerSlice)):
            lB = int(lenText * centerSlice-sliceRadius)
            rB = int(lenText * centerSlice+sliceRadius)
        else:
            lB = 0
            rB = lenText

        s = []
        for word in text_preps.split()[lB:rB]:
            if (len(word) >= 2) and not (word.isdigit()):
                prep_word = lemmatizer.parse(word)[0].normal_form
                if prep_word not in stops:
                    s.append(prep_word)
        out = ' '.join(s)
    return out


def saveData(data: object,
             filename: str) -> None:
    _, extension = os.path.splitext(filename)
    if extension == '.csv':
        data.to_csv(filename)
    elif extension == '.pkl':
        with open(filename, 'wb') as saveFile:
            pickle.dump(data, saveFile)
    elif extension == '.txt':
        with open(filename, 'w', encoding='utf-8') as file:
            file.writelines([x+'\n' for x in data])
    else:
        assert False, 'Specified wrong file extension!'

    print(f"Файл сохранен в {filename}")


def loadData(filename: str):
    assert os.path.exists(filename), 'Specified file doesnt exist!'
    _, extension = os.path.splitext(filename)
    if extension == '.csv':
        data = pd.read_csv(filename, index_col=0)
    elif extension == '.pkl':
        with open(filename, 'rb') as saveFile:
            data = pickle.load(saveFile)
    elif extension == '.txt':
        with open(filename, 'r', encoding='utf-8') as file:
            data = file.readlines()
            data = [x[1:-2] for x in data]
    else:
        assert False, 'Specified wrong file extension!'
    print(f"Данные загружены c ({filename})")
    return data


