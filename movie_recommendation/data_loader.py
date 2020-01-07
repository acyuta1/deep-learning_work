import pickle
import pandas as pd
from keras.models import load_model

MODEL = load_model("hi.h5")

DF = pd.read_csv("new_df.csv")
DF["tag"] = DF["tag"].fillna("")
PICKLE_IN = open("GENRES_MAP1.pickle", "rb")
GENRES_MAP = pickle.load(PICKLE_IN)

PICKLE_IN = open("TAG_MAP1.pickle", "rb")
TAG_MAP = pickle.load(PICKLE_IN)

INPUT_LENGTH = {"movie_len":2698, "genre_len":6189}
MOVIE_NAMES = DF.title.values.tolist()

def pad(lst, width):
    """
    Pad inputs with 0's to the required length.
    """
    lst.extend([0] * (width - len(lst)))
    return lst


def result(movie_copy, most_similar):
    """
    From the predicted indices (MovieId's), the corresponding movies are extracted 
    from the dataset.
    Also the original "watched" movie rows are extracted for comparison purposes.
    We return two dataframes: Watched and the Recommendations.
    """

    rec_movies = DF.set_index("movieId").loc[most_similar].reset_index()    
    blankIndex=[''] * len(rec_movies)
    rec_movies.index=blankIndex

    watched_movies = DF.set_index("movieId").loc[movie_copy].reset_index()
    blankIndex=[''] * len(watched_movies)
    watched_movies.index=blankIndex
    return watched_movies, rec_movies
