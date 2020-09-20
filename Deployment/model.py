import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


dataset = pd.read_csv('E:\Deployment\hiring.csv')


dataset['experience'].fillna(0,inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)


X=dataset.iloc[:,:3]

def convert_to_int(word):
    word_dict=({'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
                'eight':8,'nine':9,
                'ten':10,'eleven':11,'twelve':12,'thirteen':13,'forteen':14,
                'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,
                'twenty':20,'twenty one':21,'twenty two':22,'twenty three':23,'twenty four':24,
                'twenty five':25,'twenty six':26,'twenty seven':27,'twenty eight':28,'twenty nine':29,
                'thirty':30,'zero':0,0:0})
    return word_dict[word]



X['experience']=X['experience'].apply(lambda x: convert_to_int(x))

y=dataset.iloc[:,-1]

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(X,y)

pickle.dump(lin_reg,open('timble.pkl','wb'))

#to compare the result
model=pickle.load(open('timble.pkl','rb'))
model.predict([[2,3,5]])
