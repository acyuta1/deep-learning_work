from keras import backend as K
import pandas as pd
from keras.models import load_model
from numpy import loadtxt
import pickle

DF = pd.read_csv("df_final.csv")
DF["tag"] = DF["tag"].fillna("")
INPUT_LENGTH = {"movie_len":18, "genre_len":18}
GENRES_TOKENS = loadtxt('genres_tokens.csv', delimiter=',') 
MOVIE_NAMES = DF.title.values.tolist()

with open('movies_list', 'rb') as fp:
    MOVIES_LIST = pickle.load(fp)

MODEL1 = load_model("model_u_t_u.h5")
MODEL2 = load_model("model_i_t_i.h5")

def pad(lst, width):
    """
    Pad inputs with 0's to the required length.
    """
    lst.extend([0] * (width - len(lst)))
    return lst


def result(movie_copy, most_similar):
    """
    From the prediction indices, the corresponding movies are extracted 
    from the dataset.
    Also the original "watched" movie information is obtained.
    We return two dataframes: Watched and the Recommendations.
    """

    rec_movies = DF.set_index("movieId").loc[most_similar].reset_index()
    rec_movies = rec_movies.iloc[:,:-1]
    rec_movies = rec_movies[~rec_movies["movieId"].isin(movie_copy)]
    rec_movies = rec_movies[:40]
    watched_movies = DF.set_index("movieId").loc[movie_copy].reset_index()
    watched_movies = watched_movies.iloc[:,1:]
    return watched_movies, rec_movies
