import pandas as pd
import pickle
from keras.models import load_model

model = load_model('final_train1.h5')

df = pd.read_csv('new_df.csv')
df['tag'] = df['tag'].fillna("")
pickle_in = open("genres_map.pickle", "rb")
genres_map = pickle.load(pickle_in)

pickle_in = open("tag_map.pickle", "rb")
tag_map = pickle.load(pickle_in)


def result(movie_copy, most_similar):

    recommendations_list, watched_list = [], []

    for i in movie_copy:
        watched_list.append(df.loc[df['movieId'] == i].values[0])
    watched_movies = pd.DataFrame(
        watched_list, columns=['movieId', 'Title', 'Genres', 'Tags'])

    for i in most_similar:
        if i not in movie_copy:
            recommendations_list.append(df.loc[df['movieId'] == i].values[0])
    rec_movies = pd.DataFrame(recommendations_list, columns=[
                              'movieId', 'Title', 'Genres', 'Tags'])

    return watched_movies, rec_movies
