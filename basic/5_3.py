import numpy as np
import pandas as pd
from surprise import BaselineOnly
from surprise import KNNWithMeans
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

# MovieLens 100k 데이터 불러오기
# data = Dataset.load_builtin('ml-100k')

# Custom Dataset(Data Frame)
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', sep='\t', names=r_cols, encoding='latin-1')
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['user_id','movie_id','rating']], reader=reader)

#### (1) Grid Search
# KNN의 다양한 parameter 비교
from surprise.model_selection import GridSearchCV
param_grid = {'k':[5,10,15,25],
              'sim_options':{'name':['pearson_baseline','cosine'],
                             'user_based':[True, False]}}
gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'], cv=4)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

#### (2) Grid Search
# MF의 다양한 parameter 비교
from surprise.model_selection import GridSearchCV
param_grid = {'n_epochs':[70,80,90],
              'lr_all':[0.005,0.006,0.007],
              'reg_all':[0.05,0.07,0.1]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=4)
gs.fit(data)
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])