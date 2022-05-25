# Load Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
PATH = 'C:/RecoSys/Data/'
r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv(PATH+'u.data', sep='\t', names=r_cols, encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)

ratings_train, ratings_test = train_test_split(ratings, train_size=0.75, shuffle=True)
print(ratings_train.shape, ratings_test.shape)


# Defining RMSE measure
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# Dummy recommender 0
import random
def recommender0(recomm_list):
    recommendations = []
    for pair in recomm_list:
        recommendations.append(random.random()*4+1)
    return np.array(recommendations)

# Dummy recommender 1
def recommender1(recomm_list):
    recommendations = []
    for pair in recomm_list:
        recommendations.append(random.random()*4+1)
    return np.array(recommendations)

# 정확도(RMSE)를 계산하는 함수 
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# Hybrid 결과 얻기
weight = [0.8, 0.2]
recomm_list = np.array(ratings_test)
predictions0 = recommender0(recomm_list)
predictions1 = recommender1(recomm_list)
predictions = predictions0*weight[0] + predictions1*weight[1]
RMSE2(recomm_list[:,2], predictions)