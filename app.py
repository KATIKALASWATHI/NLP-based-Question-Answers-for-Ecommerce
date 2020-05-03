from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer

import pickle
import os
os.getcwd()


# load the model from disk
cv=pickle.load(open('transform.pkl','rb'))

clf = pickle.load(open('nlp_model.pkl', 'rb'))

app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		message = request.form['question']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    from class_def import document
    app.run(debug=True, use_reloader=False)