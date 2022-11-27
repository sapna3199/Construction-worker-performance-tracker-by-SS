# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import pickle

app = Flask(__name__)

model = pickle.load(open('rf_model.pkl', 'rb'))

df = pd.DataFrame()

@app.route('/')
def home():
    return(render_template('index.html'))

@app.route('/predict',methods=['POST'])
def predict():
    
    # input
    input_features = [float(x) for x in request.form.values()]
    feature_value = np.array(input_features)
    
    # output
    output = model.predict([feature_value])[0].round(0)
    

    return render_template('index.html', prediction_text='Performance rating is [{}] '.format(output))
  

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
