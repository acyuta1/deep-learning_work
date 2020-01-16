from prediction_helper import prepare_inputs, make_prediction
from flask import Flask, render_template, request
from data_loader import MODEL1, MODEL2, result, MOVIE_NAMES

application = Flask(__name__)
@application.route("/")

def home():
    return render_template("home.html", data = MOVIE_NAMES)

@application.route("/predict", methods=["POST"])
def predict():

    if request.method == "POST":

        movies = request.form.getlist('mymultiselect')
        if(len(movies)<7):
            g_input, movie_input, movie_copy = prepare_inputs(movies,1)
            preds = make_prediction(g_input, MODEL2, movie_input)
            watched_movies, rec_movies = result(movie_copy, preds)
        else:
            movies, g_input, movie_copy = prepare_inputs(movies,2)
            print(movies[:10],g_input[:10])
            preds = make_prediction(movies, g_input, MODEL1, movie_copy)
            watched_movies, rec_movies = result(movie_copy, preds)

        context = {
            "watched_movies_table": [watched_movies.to_html(classes="data")],
            "watched_movies_titles": watched_movies.columns.values,
            "rec_movies_table": [rec_movies.to_html(classes="data")],
            "rec_movies_titles": rec_movies.columns.values,
        }
    return render_template("prediction.html", context=context)

if __name__ == "__main__":
    global MODEL
    MODEL1 = MODEL1
    MODEL2 = MODEL2
    MOVIE_NAMES = MOVIE_NAMES
    application.run(threaded=False)
