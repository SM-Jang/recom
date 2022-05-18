import pandas as pd

### Load Data ###
path = 'C:/RecoSys/Data/'

# u.user 파일을 DataFrame으로 읽기
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(path+'u.user', sep='|', names=u_cols, encoding='latin-1')

# u.item 파일을 DataFrame으로 읽기
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL',
         'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv(path+'/u.item', sep='|', names=i_cols, encoding='latin-1')

# u.data 파일을 DataFrame으로 읽기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(path+'u.data', sep='\t', names=r_cols, encoding='latin-1')


# timestamp 제거
ratings = ratings.drop('timestamp', axis=1)

# movie ID와 title 뺴고 다른 데이터 제거
movies = movies[['movie_id', 'title']]


### Train/Test split ###
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)



### Recommendation ###
import numpy as np
def RMSE(y_true, y_pred):
    '''RMSE 계산 함수'''
    error = np.array(y_true)-np.array(y_pred)
    return np.sqrt(np.mean(error**2))

def score(model):
    '''추천 방법에 따른 성능 계산 함수'''
    id_pairs = zip(x_test['user_id'], x_test['movie_id'])
    y_pred = np.array( [model(user, movie) for user, movie in id_pairs] )
    y_true = np.array( x_test['rating'] )
    return RMSE(y_true, y_pred)

def best_seller(user_id, movie_id):
    ''' 평점이 가장 높은 영화 추천'''
    try:
        rating = train_mean['movie_id']
    except:
        rating = 3.0
    return rating


def cf_gender(user_id, movie_id):
    '''Gender별 평균을 예측치로 돌려주는 함수'''
    
    rating = 3.0
    if movie_id in rating_matrix:
        gender = users.loc[user_id, 'sex']
        if gender in g_mean[movie_id]:
            rating = g_mean[movie_id][gender]
    return rating

def cf_occupation(user_id, movie_id):
    '''Occupation별 평균을 예측치로 돌려주는 함수'''
    
    rating = 3.0
    if movie_id in rating_matrix:
        occupation = users.loc[user_id, 'occupation']
        if occupation in o_mean[movie_id]:
            rating = o_mean[movie_id][occupation]
    return rating

def cf_occ_gend(user_id, movie_id):
    '''Occupation과 Gender별 평균을 예측치로 돌려주는 함수'''
    
    rating = 3.0
    if movie_id in rating_matrix:
        occupation = users.loc[user_id, 'occupation']
        if occupation in g_o_mean[movie_id]:
            gender = users.loc[user_id, 'sex']
            if gender in g_o_mean[movie_id][occupation]:
                rating = g_o_mean[movie_id][occupation][gender]
    return rating

train_mean = x_train.groupby(['movie_id'])['rating'].mean()
print(f'The best-seller model score is {score(best_seller)}')


rating_matrix = x_train.pivot(index='user_id', columns='movie_id', values='rating')


# Full Matrix
merged_ratings = pd.merge(x_train, users)
users = users.set_index('user_id')

# CF: Gender 
g_mean = merged_ratings[['movie_id', 'sex', 'rating']].groupby(['movie_id', 'sex'])['rating'].mean()
print(f'The CF score by gender is {score(cf_gender)}')

# CF: occupation
o_mean = merged_ratings[['movie_id', 'occupation', 'rating']].groupby(['movie_id', 'occupation'])['rating'].mean()
print(f'The CF score by occupation is {score(cf_occupation)}')

# CF: Occ and Gender
g_o_mean = merged_ratings[['movie_id', 'occupation', 'sex', 'rating']].groupby(['movie_id', 'occupation', 'sex'])['rating'].mean()
print(f'The CF score by occupation and gender is {score(cf_occ_gend)}')