import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# For LSTM Model
# from keras.models import load_model
# from keras_preprocessing.sequence import pad_sequences

# For SVC and Random Forest Model
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create flask app
flask_app = Flask(__name__)

# Load the Model
#model = pickle.load(open("models/randomforest_model.pkl", "rb")) 
model = pickle.load(open("models/svm_model.pkl", "rb"))
#model = load_model('models/lstm_model.h5')


# For LSTM
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# For SVM and Random Forest
with open('models/vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)

# Function to predict sentiment from the entered text
def predict_sentiment(text,model_name):


	sentiment_classes = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
	#model_name = model.__class__.__name__

	if model_name == "SVC" or "RandomForestClassifier":
		text_vector = vectorizer.transform([text])
		predict = model.predict(text_vector)
	# LSTM model
	else:
		max_len=50
		# Transforms text to a sequence of integers using a tokenizer object
		xt = tokenizer.texts_to_sequences(text)
	    # Pad sequences to the same length
		xt = pad_sequences(xt, padding='post', maxlen=max_len)
	    # Do the prediction using the loaded model
		predict = model.predict(xt).argmax(axis=1)

	return sentiment_classes[predict[0]]


@flask_app.route("/")
def Home():
    return render_template("index.html")


# # For SVM and Random Forest Model

@flask_app.route("/predict", methods = ["POST"])
def predict():

	# tokenizer = TweetTokenizer()
	# vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)

	# Text entered to be Predicted

	# print (model.__class__.__name__)
    model_name = model.__class__.__name__
    if model_name == "SVC" or "RandomForestClassifier":
    	text = request.form.get("text")
    else:
    	text = request.form.values()

    sentiment = predict_sentiment(text,model_name)
    return render_template("index.html", prediction_text = "Sentiment Prediction : {}".format(sentiment))

	# sentiment_classes = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

	# # Vectorize the Text
	# text_vector = vectorizer.transform([text])
	# # Predict
	# predict = model.predict(text_vector)

	# Check for Prediction Probability
	# TODO



	

if __name__ == "__main__":
    flask_app.run(debug=True)



