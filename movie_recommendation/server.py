from predict import prepare_inputs, make_prediction
from data_loader import MODEL, result, MOVIE_NAMES
from flask import Flask, render_template, request

application = Flask(__name__)


@application.route("/")
def home():
    return render_template("home.html", data = MOVIE_NAMES)


@application.route("/predict", methods=["POST"])
def predict():

    if request.method == "POST":

        movies = request.form.getlist('mymultiselect')        
        movies, g_input, movie_copy = prepare_inputs(movies)

        preds = make_prediction(movies, g_input, MODEL)

        most_similar = preds[0].argsort()[-(10 + len(movie_copy)) :][::-1]

        watched_movies, rec_movies = result(movie_copy, most_similar)

        context = {
            "watched_movies_table": [watched_movies.to_html(classes="data")],
            "watched_movies_titles": watched_movies.columns.values,
            "rec_movies_table": [rec_movies.to_html(classes="data")],
            "rec_movies_titles": rec_movies.columns.values,
        }

    return render_template("prediction.html", context=context)

if __name__ == "__main__":
    global MODEL
    MODEL = MODEL
    global MOVIE_NAMES
    MOVIE_NAMES = MOVIE_NAMES
    application.run(threaded=False)
