from keras import backend as K
import pandas as pd
from keras.models import load_model
from numpy import loadtxt


DF = pd.read_csv("df_final.csv")
DF["tag"] = DF["tag"].fillna("")
GENRES_TOKENS = loadtxt('genres_tokens.csv', delimiter=',') 
INPUT_LENGTH = {"movie_len":36, "genre_len":36}
MOVIE_NAMES = DF.title.values.tolist()

def full_multi_label_metric(y_true, y_pred):
    comp = K.equal(y_true, K.round(y_pred))
    return K.cast(K.all(comp, axis=-1), K.floatx())

MODEL1 = load_model("done.h5",custom_objects={"full_multi_label_metric": full_multi_label_metric})
MODEL2 = load_model("single_withoutopt.h5")

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
    print("here",most_similar)

    rec_movies = DF.set_index("movieId").loc[most_similar].reset_index()
    rec_movies = rec_movies.iloc[:,:-1]
    rec_movies = rec_movies[~rec_movies["movieId"].isin(movie_copy)]
    watched_movies = DF.set_index("movieId").loc[movie_copy].reset_index()
    watched_movies = watched_movies.iloc[:,1:]
    return watched_movies, rec_movies
