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

# Training models
algorithms = [BaselineOnly, KNNWithMeans, SVD, SVDpp]
names = []
results = []
for option in algorithms:
    algo = option()
    names.append(option.__name__)
    algo.fit(trainset)
    predictions = algo.test(testset)
    results.append(accuracy.rmse(predictions))

# 성능 시각화
import matplotlib.pyplot as plt
names = np.array(names)
results = np.array(results)

index = np.argsort(results)
plt.figure(figsize=(8,5))
plt.title('Accuracy[RMSE]: ML-100k')
plt.ylim(0.8, 1)
plt.bar(names[index], results[index])
plt.savefig('Surprise using ml-100k.jpg')
plt.show()