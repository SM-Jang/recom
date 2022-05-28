import numpy as np
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
data = Dataset.load_builtin('ml-100k')

# Train / Test 분리
trainset, testset = train_test_split(data, test_size=0.25)

#### (1)
# 정확도 계산
algo = KNNWithMeans()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)