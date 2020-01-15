from keras import backend as K
import pandas as pd
from keras.models import load_model
from numpy import loadtxt
import numpy as np
import pickle

DF = pd.read_csv("df_final.csv")
DF["tag"] = DF["tag"].fillna("")
GENRES_TOKENS = loadtxt('genres_tokens.csv', delimiter=',') 
INPUT_LENGTH = {"movie_len":18, "genre_len":18}
MOVIE_NAMES = DF.title.values.tolist()

with open('genre_movie_dict.pickle', 'rb') as fp:
    GENRE_MOVIE_DICT = pickle.load(fp)

def sample(preds, temperature=0.1):
    '''
    Reweighting the distribution to a certain "temperature".
    Sampling the next movie at random according to the reweighted distribution
    Adding the new movie at the end of the available sequence prediction.
    '''
    preds = np.asarray(preds).astype('float64')
    exp_preds = preds - np.exp(temperature)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    argmax = np.argmax(probas)
    return [argmax,GENRE_MOVIE_DICT[argmax]]

MODEL1 = load_model("generated_seed_without_opt.h5")
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
