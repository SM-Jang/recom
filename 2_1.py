import pandas as pd
import os

PATH = 'C:/RecoSys/Data/'
os.listdir(PATH)

# load u.user file as DataFrame
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(PATH+'u.user', sep='|', names=u_cols, encoding='latin-1')
users = users.set_index('user_id')
users.head()


# load u.item file as DataFrame
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL',
         'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv(PATH+'u.item', sep='|', names=i_cols, encoding='latin-1')
movies.set_index('movie_id')
movies.head()


# load u.data file as DataFrame
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(PATH+'u.data', sep='\t', names=r_cols, encoding='latin-1')
ratings.head()



## Best-seller recommendation ##
def recom_movie(n_itmes):
    '''평점이 가장 높은 상품 (best-item) 추천'''
    
    top_n_movie_sort = movie_mean.sort_values(ascending=False)[:5]
    recommendations = movies.loc[top_n_movie_sort.index, 'title']
    return recommendations
    
movie_mean = ratings.groupby(['movie_id'])['rating'].mean()

n=5
print(f'The top-{n} rating movies are \n{recom_movie(n)}')
