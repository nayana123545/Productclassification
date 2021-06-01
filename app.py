from django.http.response import HttpResponseRedirect
from flask import Flask,render_template,request,url_for
from flask.templating import render_template_string
#import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.externals import joblib
from django.http import HttpResponse
from sklearn.preprocessing import StandardScaler
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer


clf=pickle.load(open('product_classification.pkl','rb'))
cv=pickle.load(open('transform.pkl','rb'))


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        if not message:
            my_prediction="Please enter the title"  
            return render_template('index.html',my_prediction=my_prediction)       
        else:
            data=[message]
            vect=cv.transform(data).toarray()
            my_prediction=clf.predict(vect)
        
    return render_template('result.html',prediction=my_prediction,message=message)      


if __name__=="__main__":
    app.run(debug=True)
