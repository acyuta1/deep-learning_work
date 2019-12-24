# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import numpy as np
from data_loader import model, result
from prediction_helper import prepare_inputs


app = Flask(__name__)
model = model


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':

        comment = request.form['comment']
        movies = [int(x) for x in comment.split(',')]
        movies = list(set(movies))

        movies, g_input, t_input, movie_copy = prepare_inputs(movies)

        print(movies, t_input, g_input)

        preds = model.predict(
            [np.array([movies, ]), np.array([g_input, ]), np.array([t_input, ])])
        most_similar = preds[0].argsort()[-(10+len(movie_copy)):][::-1]

        watched_movies, rec_movies = result(movie_copy, most_similar)

        context = {"table1": [watched_movies.to_html(classes='data')],
                   "title1": watched_movies.columns.values,
                   "table2": [rec_movies.to_html(classes='data')],
                   "title2": rec_movies.columns.values
                   }

    return render_template('prediction.html', context=context)


if __name__ == '__main__':
    app.run(threaded=False)
