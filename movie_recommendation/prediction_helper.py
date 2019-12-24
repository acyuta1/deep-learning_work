# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:56:03 2019

@author: Achyutha.aluru
"""

from data_loader import df,genres_map,tag_map
import numpy as np

def prepare_inputs(movie_input):
    
    movies_copy = movie_input
    
    genres = []
    tags = []
    genre_input = []
    tag_input = []
    for i in movie_input:
            genres.append(df.loc[df['movieId'] == i, 'genres'].iloc[0])
            tags.append(df.loc[df['movieId'] == i, 'tag'].iloc[0])
    
    for i in genres:
        for j in i.split():
            if '-' in j:
                genre_input.append(list(genres_map.keys())[list(genres_map.values()).index(j.split('-')[0])])
                genre_input.append(list(genres_map.keys())[list(genres_map.values()).index(j.split('-')[1])])
            else:
                genre_input.append(list(genres_map.keys())[list(genres_map.values()).index(j)])
    genre_input = list(set(genre_input))
    N = 24 - len(genre_input)
    genre_input = np.pad(genre_input, (0, N), 'constant')
    
    
    for i in tags:
        for j in i.split():
            if '-' in j:
                tag_input.append(list(tag_map.keys())[list(tag_map.values()).index(j.split('-')[0])])
                tag_input.append(list(tag_map.keys())[list(tag_map.values()).index(j.split('-')[1])])
            elif '!' in j:
                tag_input.append(list(tag_map.keys())[list(tag_map.values()).index(j.split('!')[0])])
            else:
                tag_input.append(list(tag_map.keys())[list(tag_map.values()).index(j)])
    tag_input = list(set(tag_input))
    N = 100 - len(tag_input)
    tag_input = np.pad(tag_input, (0, N), 'constant')
    
    N = 2698 - len(movie_input) 
    movie_input = np.pad(movie_input, (0, N), 'constant')
        
    return movie_input, genre_input, tag_input, movies_copy
