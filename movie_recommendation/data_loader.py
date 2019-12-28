import pickle
import pandas as pd
from keras.models import load_model

MODEL = load_model('final_train1.h5')

DF = pd.read_csv('new_df.csv')
DF['tag'] = DF['tag'].fillna("")
PICKLE_IN = open("genres_map.pickle", "rb")
GENRES_MAP = pickle.load(PICKLE_IN)

PICKLE_IN = open("tag_map.pickle", "rb")
TAG_MAP = pickle.load(PICKLE_IN)


def result(movie_copy, most_similar):
    """
    From the prediction indices, the corresponding movies are extracted 
    from the dataset.
    Also the original "watched" movie information is obtained.
    We return two dataframes: Watched and the Recommendations.
    """
    recommendations_list, watched_list = [], []

    for i in movie_copy:
        watched_list.append(DF.loc[DF['movieId'] == i].values[0])
    watched_movies = pd.DataFrame(
        watched_list, columns=['movieId', 'Title', 'Genres', 'Tags'])

    for i in most_similar:
        if i not in movie_copy:
            recommendations_list.append(DF.loc[DF['movieId'] == i].values[0])
    rec_movies = pd.DataFrame(recommendations_list, columns=[
                              'movieId', 'Title', 'Genres', 'Tags'])

    return watched_movies, rec_movies
