# In[1]
import re
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
#from flask_ngrok import run_with_ngrok 

# In[2]
 
app = Flask(__name__)
#run_with_ngrok(app)
#filename = 'C:/Users/HARSHA/Documents/Jigsaw_Comment_classification/trained_model.pkl'
xgb_loaded = pickle.load(open('trained_model.pkl', 'rb'))
cv = pickle.load(open('tranform.pkl','rb'))

# In[3]

def get_input(inp):
    char = ""
    for w in inp:
        char = w + " " + char
    words = char.lower()
    words_w = words.strip()
    words_c = re.sub('[^a-zA-Z]', '', words_w)
    s = words_c
    df = pd.DataFrame(data = {'comment_text' : s}, index=[0])
    X = cv.transform(df['comment_text'])
    return X
    
# In[4]
    
@app.route('/',endpoint='home') 
def home():
    return render_template('index.html')

# In[5]
    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    inp_text = [str(x) for x in request.form.values()]
    X=get_input(inp_text)
    output = xgb_loaded.predict_proba(X)
    d={} 
    d['toxic']=output[0][0]
    d['severe toxic']=output[0][1]
    d['obscene']=output[0][2]
    d['threat']=output[0][3]
    d['insult']=output[0][4]
    d['identity hate']=output[0][5]
    #dc={}
    #dc['toxic']=preds[0][0]
    #dc['severe toxic']=preds[0][1]
    #dc['obscene']=preds[0][2]
    #dc['threat']=preds[0][3]
    #dc['insult']=preds[0][4]
    #dc['identity hate']=preds[0][5]


    return render_template('index.html', prediction_probabilities='Prediction probabilities are {}'.format(d))

# In[6]
    
if __name__ == "__main__":
    app.run() 

# In[7]
    
