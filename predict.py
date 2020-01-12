import numpy as np
from data_loader import DF, pad, INPUT_LENGTH


def make_prediction(movies, g_input, model):
    """
    Function to predict provided we have the processed inputs. 
    """
    return model.predict(
        [np.array([movies,]), np.array([g_input,])]
    )


def prepare_inputs(movie_input):
    """
    With only MovieID's as the input to our model, we are required to 
    obtain the required genre's and tag's and bring our raw input to
    the required shape.
    """
    
    movie_input = [DF.movieId[DF['title']==i].iloc[0] for i in movie_input]
    movies_copy = movie_input[:]

    genres = (DF.genres_class[DF["movieId"].isin(movie_input)].values).tolist()

    genres = pad(genres, INPUT_LENGTH["genre_len"])    
    movie_input = pad(movie_input, INPUT_LENGTH["movie_len"])
    
    return movie_input, genres, movies_copy
