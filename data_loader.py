import pandas as pd
from keras.models import load_model

MODEL = load_model("victory1.h5")

DF = pd.read_csv("df_final.csv")
DF["tag"] = DF["tag"].fillna("")

INPUT_LENGTH = {"movie_len":36, "genre_len":36}
MOVIE_NAMES = DF.title.values.tolist()

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
    watched_movies = DF.set_index("movieId").loc[movie_copy].reset_index()
    watched_movies = watched_movies.iloc[:,1:]
    return watched_movies, rec_movies
