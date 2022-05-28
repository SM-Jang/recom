import numpy as np
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', sep='\t', names=r_cols, encoding='latin-1')

ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)

class NEW_MF():
    def __init__(self, ratings, K, alpha, beta, iterations, verbose=True):
        self.R = np.array(ratings)
        #### (2) user_id, item_id를 R의 index와 매핑하기 위한 dictionary 생성 ####
        item_id_index = []
        index_item_id = []
        for i, one_id in enumerate(ratings):
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])
        self.item_id_index = dict(item_id_index)
        self.index_item_id = dict(index_item_id)        
        user_id_index = []
        index_user_id = []
        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)
        #### <<<< (2)
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose
        
    def rmse(self):
        '''train set의 RMSE 계산'''
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []
        for x, y in zip(xs,ys):
            prediction = self.get_prediction(x,y)
            self.predictions.append(prediction)
            self.errors.append(self.R[x,y] - prediction)
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))
    
    def get_prediction(self, i, j):
        '''Ratings for user i and item j'''
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i,:].dot(self.Q[j,:].T)
        return prediction
    
    def sgd(self):
        # Stochastic gradient descent to get optimized P and Q matrix
        for i, j, r in self.samples:
            prediction = self.get_prediction(i,j)
            e = r - prediction
            
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
            
### >>>> (3)
    # Test set을 선정
    def set_test(self, ratings_test):
        '''분리된 test set을 넘겨받아서 클래스 내부의 test set을 만드는 함수'''
        test_set = []
        for i in range(len(ratings_test)):
            
            x = self.user_id_index[ratings_test.iloc[i,0]] # user_id -> index
            y = self.item_id_index[ratings_test.iloc[i,1]] # movie_id -> index
            z = ratings_test.iloc[i,2] # rating(GT)
            test_set.append([x,y,z]) # 사용자, 아이템, 평점을 test_set에 추가
            self.R[x, y] = 0 # 해당 (사용자-아이템-평점)을 R에서 0으로 지운다 - MF모델을 학습하기 위해
        self.test_set = test_set
        return test_set
    
    def test_rmse(self):
        '''Test set의 RMSE 계산'''
        error = 0
        for one_set in self.test_set: # 0:사용자, 1:아이템, 2:평점
            predicted = self.get_prediction(one_set[0], one_set[1])
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error/len(self.test_set))
    
    def test(self):
        # Initialize user_feature and item_feature matrix
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])
        
        # List of training samples
        rows, columns = self.R.nonzero()
        self.samples = [(i, j, self.R[i,j]) for i, j in zip(rows, columns)]
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse1 = self.rmse()
            rmse2 = self.test_rmse()
            training_process.append((i+1, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print('Iteration: %d ; Train RMSE = %.4f ; Test RMSE = %.4f'
                          % (i+1, rmse1, rmse2))
                    
        return training_process
    
    def get_one_prediction(self, user_id, item_id):
        return self.get_prediction(self.user_id_index[user_id], self.item_id_index[item_id])
    
    def full_prediction(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_d[np.newaxis, :] + self.P.dot(self.Q.T)
    

        
        
#### (1)
# train test 분리
from sklearn.utils import shuffle
TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state=1)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

R_temp = ratings.pivot_table(values='rating', index='user_id', columns='movie_id').fillna(0)
mf = NEW_MF(R_temp, K=30, alpha=0.001, beta=0.02, iterations=100, verbose=True)
test_set = mf.set_test(ratings_test)
result = mf.test()

# Printing predictions
print(mf.full_prediction())
print(mf.get_one_prediction(1, 2))