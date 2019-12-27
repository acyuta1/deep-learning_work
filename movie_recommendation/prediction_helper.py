import numpy as np
from data_loader import DF, GENRES_MAP, TAG_MAP

def make_prediction(movies, g_input, t_input, model):
    '''
    Function to predict provided we have the processed inputs. 
    '''
    return model.predict(
            [np.array([movies, ]), np.array([g_input, ]), np.array([t_input, ])])


def prepare_inputs(movie_input):
    '''
    With only MovieID's as the input to our model, we are required to 
    obtain the required genre's and tag's and bring our raw input to
    the required shape.
    '''
    movies_copy = movie_input

    genres, tags, genre_input, tag_input = [], [], [], []
    
    for i in movie_input:
        genres.append(DF.loc[DF['movieId'] == i, 'genres'].iloc[0])
        tags.append(DF.loc[DF['movieId'] == i, 'tag'].iloc[0])

    for i in genres:
        for j in i.split():
            genre_input.append(list(GENRES_MAP.keys())[
                                   list(GENRES_MAP.values()).index(j)])
    
    genre_input = list(set(genre_input))
    required_length = 24 - len(genre_input)
    genre_input = np.pad(genre_input, (0, required_length), 'constant')

    for i in tags:
        for j in i.split():
            if j in ('em','cs','se'):
                continue
            else:
                tag_input.append(list(TAG_MAP.keys())[
                                 list(TAG_MAP.values()).index(j)])
    
    tag_input = list(set(tag_input))
    
    if len(tag_input) > 100:
        tag_input = tag_input[0:100]
    else:
        required_length = 100 - len(tag_input)
        tag_input = np.pad(tag_input, (0, required_length), 'constant')

    required_length = 2698 - len(movie_input)
    movie_input = np.pad(movie_input, (0, required_length), 'constant')

    return movie_input, genre_input, tag_input, movies_copy
