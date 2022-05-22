import numpy as np
import pandas as pd

#### (1)
# MF class
class MF():
    def __init__(self, ratings, K, alpha, beta, iterations, verbose=True):
        self.R = np.array(ratings) # 실제 평점
        self.num_users, self.num_items = np.shape(self.R) # 사용자 수, 아이템 수
        self.K = K # 잠재 요인의 수
        self.alpha = alpha # learning_rate
        self.beta = beta # regularization
        self.iterations = iterations # 반복 횟수
        self.verbose = verbose # 학습과정 출력
        
    def rmse(self):
        '''현재 P,Q를 가지고 RMSE를 계산해 주는 함수'''
        xs ,ys = self.R.nonzero() # ratings에서 0이 아닌 index(x, y)
        self.predictions = []
        self.errors = []
        for x, y in zip(xs, ys): # 평점이 있는 요소들 중에서
            prediction = self.get_prediction(x,y) # 사용자x, 아이템 y 각각에 대해서 평점 예측
            self.predictions.append(prediction)
            self.errors.append(self.R[x,y] - prediction) # 실제 값과 예측 값의 차이
        self.predictions = np.array(self.predictions) # list -> numpy array
        self.errors = np.array(self.errors) # list -> numpy array
        return np.sqrt(np.mean(self.errors**2))
    
    def train(self):
        '''
        정해진 반복 횟수(self.iterations)만큼 
        앞의 식(2), (4)를 사용해 
        P, Q, bu, bd값을 update하는 함수
        '''
        # Initialize user-feature(P) and movie-feature(Q) matrix
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K)) # 평균:0, 표준편차:1/K
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K)) # 평균:0, 표준편차:1/K
        
        # Initalize the bias terms
        self.b_u = np.zeros(self.num_users) # user의 평가경향을 0으로 초기화
        self.b_d = np.zeros(self.num_items) # items의 평가경향
        self.b = np.mean(self.R[self.R.nonzero()]) # 전체 평균
        
        # List of training samples
        rows, columns = self.R.nonzero() # 평점이 있는 위치
        # SGD를 적용할 대상, 즉 평점이 있는 요소의 인덱스와 평점을 sample
        self.samples = [(i,j,self.R[i,j]) for i, j in zip(rows, columns)]
        
        
        # Stochastic Gradient Descent for given number pf iterations
        training_process = [] # SGD를 한 번 실행할 때마다 RMSE가 얼마나 개선되는지 기록
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd() # SGD 실행
            rmse = self.rmse() # SGD로 P, Q, b_u, b_d 업데이트 후 RMSE 계산
            training_process.append((i+1, rmse)) # 결과 저장
            if self.verbose:
                if (i+1)%10==0:
                    print("Iteration: %d ; Train RMSE = %.4f" %(i+1, rmse))
        return training_process
    
    # Rating prediction for user i and item j
    def get_prediction(self, i, j):
        '''평점 예측값을 구하는 함수'''
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i,:].dot(self.Q[j,:].T)
        return prediction
    
    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)
            
            # calculate gradient
            self.b_u[i] = self.b_u[i] + self.alpha * (e - self.beta*self.b_u[i])
            self.b_d[j] = self.b_d[j] + self.alpha * (e - self.beta*self.b_d[j])
            
            # update
            self.P[i, :] += self.alpha * (e * self.Q[j,:] - self.beta*self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i,:] - self.beta*self.Q[j,:])
            
    
        


r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', sep='\t', names=r_cols, encoding='latin-1')

ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)

# 전체 데이터 사용 MF
R_temp = ratings.pivot_table(values='rating', index='user_id', columns='movie_id').fillna(0)
mf = MF(R_temp, K=30, alpha=0.001, beta=0.02, iterations=100, verbose=True)
train_process = mf.train()
