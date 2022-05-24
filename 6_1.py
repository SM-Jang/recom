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

#### (1)
# SVD(MF) using DL
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adamax

# Variable 초기화
K = 200                     # latent factor 수
mu = ratings_train.rating.mean()   # 전체 평균
M = ratings.user_id.max()+1 # Number of users
N = ratings.movie_id.max()+1# Number of items

# Defining RMSE measure
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))

# Keras Model
user = Input(shape=(1,))
item = Input(shape=(1,))
P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user) # P: [M, K]
user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user) # user bias: [M, 1]

Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item) # Q: [N, K]
item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item) # item bias: [N, 1]

R = layers.dot([P_embedding, Q_embedding], axes=2)
R = layers.add([R, user_bias, item_bias])
R = Flatten()(R)

# Model Setting
model = Model(inputs = [user, item], outputs = R)
model.compile(
    loss = RMSE,
    optimizer=SGD(),
    metrics=[RMSE]
)
model.summary()

# Model Fitting
result = model.fit(
    x = [ratings_train.user_id.values, ratings_train.movie_id.values],
    y = ratings_train.rating.values - mu,
    epochs = 60,
    batch_size=256,
    validation_data=(
        [ratings_test.user_id.values, ratings_test.movie_id.values],
        ratings_test.rating.values-mu
    )
)


# Plot RMSE
import matplotlib.pyplot as plt
plt.plot(result.history['RMSE'], label="Train RMSE")
plt.plot(result.history['val_RMSE'], label='Test RMSE')
plt.legend()
plt.show()


# Prediction
user_ids = ratings_test.user_id.values[:6]
movie_ids = ratings_test.movie_id.values[:6]
predictions = model.predict([user_ids, movie_ids])+mu
print('Actuals:\n', ratings_test[:6])
print()
print('Predictions:\n', predictions)


# RMSE check
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

user_ids = ratings_test.user_id.values
movie_ids = ratings_test.movie_id.values
y_pred = model.predict([user_ids, movie_ids]) + mu
y_pred = np.ravel(y_pred, order='C')
y_true = np.array(ratings_test.rating)

RMSE2(y_true, y_pred)
