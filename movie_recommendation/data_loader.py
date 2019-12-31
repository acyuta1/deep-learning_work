import pickle
import pandas as pd
from keras.models import load_model

MODEL = load_model("final_train1.h5")

DF = pd.read_csv("new_df.csv")
DF["tag"] = DF["tag"].fillna("")
PICKLE_IN = open("genres_map.pickle", "rb")
GENRES_MAP = pickle.load(PICKLE_IN)

PICKLE_IN = open("tag_map.pickle", "rb")
TAG_MAP = pickle.load(PICKLE_IN)

INPUT_LENGTH = [2698, 24, 100]  # Input length required for movies, genres 
                                # and tags

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

    watched_movies = DF.set_index("movieId").loc[movie_copy].reset_index()

    return watched_movies, rec_movies
