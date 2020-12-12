#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary packages

import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


tr_df = pd.read_csv(r"C:\Users\HARSHA\Downloads\train.csv")
ts_df = pd.read_csv(r"C:\Users\HARSHA\Downloads\test.csv")


# ## Step - 1 :

# In[3]:


tr_df


# In[4]:


ts_df


# In[5]:


tr_df.dtypes


# In[6]:


tr_df.describe()


# In[7]:


# We will know that how many rows and columns were there in the dataset in this particular cell by using "count"

rows_count, columns_count = tr_df.shape
print('Total Number of rows :', rows_count)
print('Total Number of columns :', columns_count)


# In[8]:


tr_df.info()


# In[9]:


# Now we are going to check whether the dataset contains missing values or not :

tr_df.isnull()


# In[10]:


# we can see there is no missing value in the dataframe.
# i.e False refers that the dataset does not contain the null values. You can see clearly in below section

tr_df.isnull().sum()


# #### From above we got : The observation is "NO MISSING VALUES" ;

# In[11]:


# Here the duplicated data is assigned to the variable "dup" and it shows whether the data contains duplicated data or not.

dup = tr_df.duplicated()


# In[12]:


dup


# In[13]:


# Now we drop some unnecessary columns from the dataset

tr_df = tr_df.drop(['id'], axis=1 )
ts_df = ts_df.drop(['id'], axis=1 )


# #### We removed "id" column in both training set and as well as test set as that column is not important for further process

# In[14]:


tr_df.head()


# In[15]:


ts_df.head()


# ## EDA :-

# In[16]:


tr_df.nunique()


# In[17]:


ts_df.nunique()


# In[18]:


print(tr_df.shape, ts_df.shape)


# In[19]:


tr_df["toxic"].value_counts().to_frame()


# In[20]:


tr_df["severe_toxic"].value_counts().to_frame()


# In[21]:


tr_df["obscene"].value_counts().to_frame()


# In[22]:


tr_df["threat"].value_counts().to_frame()


# In[23]:


tr_df["insult"].value_counts().to_frame()


# In[24]:


tr_df["identity_hate"].value_counts().to_frame()


# In[25]:


tr_df.corr()


# In[26]:


ts_df.corr()


# In[27]:


# Heatmap
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))
sns.heatmap(tr_df.corr(), annot = True)


# In[28]:


import matplotlib.pyplot as plt

sns.countplot(x='toxic',data=tr_df)
plt.title('Distribution of Toxic Comments')


# In[29]:


# Pairplot

sns.pairplot(tr_df.iloc[:,1:])
plt.show()


# ## Step-2 : Data Pre-Processing

# In[30]:


import nltk


# In[31]:


from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[32]:


stop_words = set(stopwords.words('english'))


# In[33]:


import warnings
warnings.filterwarnings('ignore')


# In[114]:


import contractions 
import re
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
def preprocess(char):
    words = char.lower()
    words_abb = contractions.fix(words)
    words_w = words_abb.strip()
    words_c = re.sub('[^a-zA-Z]',' ',words_w) 
    words_token = word_tokenize(words_c)
    words_clean = [w for w in words_token if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words_lemm = [lemmatizer.lemmatize(w) for w in words_clean]    
    return words_lemm


# In[35]:


for i in range(0,len(tr_df)):
    char = []
    s = ""
    char = preprocess(tr_df['comment_text'][i])  
    for w in char:
        s = s + w + " "
    tr_df['comment_text'][i] = s


# In[36]:


tr_df


# In[37]:


for i in range(0,len(ts_df)):
    char = []
    s = ""
    char = preprocess(tr_df['comment_text'][i])  
    for w in char:
        s = s + w + " "
    ts_df['comment_text'][i] = s


# In[38]:


ts_df


# ## Model Training : 

# In[39]:


from sklearn.feature_extraction.text import CountVectorizer

X = tr_df['comment_text']
Y = tr_df.drop(['comment_text'],axis=1)


# In[40]:


from sklearn.model_selection import train_test_split

Xtr,Xts,Ytr,Yts = train_test_split(X,Y,test_size = 0.3, random_state = 0) 
print(Xtr.shape,Xts.shape,Ytr.shape,Yts.shape)


# In[41]:


cv = CountVectorizer()
Xtr = cv.fit_transform(Xtr)
Xts = cv.transform(Xts)


# In[42]:


from sklearn.naive_bayes import MultinomialNB
multinomialnb = MultinomialNB()


# In[43]:


from sklearn import metrics
import itertools


# In[44]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score


# In[45]:


from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

model = OneVsRestClassifier(LogisticRegression())
model.fit(Xtr,Ytr)


# In[46]:


y_pred = model.predict(Xts)
y_pred


# In[47]:


score = metrics.accuracy_score(Yts,y_pred)
print("accuracy :  %0.3f" % score)


# In[48]:


from xgboost import XGBClassifier


# In[49]:


xgb = OneVsRestClassifier(XGBClassifier())
xgb.fit(Xtr, Ytr)


# In[50]:


y_pred = xgb.predict(Xts)
y_pred


# In[51]:


score = metrics.accuracy_score(Yts,y_pred)
print("accuracy :  %0.3f" % score)


# In[52]:


f_pred = model.predict_proba(Xts)


# In[53]:


f_pred


# In[ ]:





# In[95]:


new = pd.DataFrame(f_pred) 


# In[96]:


final_file = new.to_csv('final.csv')


# In[97]:


new.rename(columns = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}, inplace = True)


# In[98]:


new


# In[ ]:





# ## We performed two algorithms i.e XGBoost and Logistic Regression. Accuracy score for XGBoost is 91.8 and for Logistic Regression is 91.3.
# 
# ## ***As 91.8 is the best accuracy score, so XGBoost is the best algorithm.***

# In[ ]:





# In[99]:


import pickle


# In[100]:


with open('final_trained_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)


# In[101]:


with open('final_trained_model.pkl', 'rb') as f:
    xgb_loaded = pickle.load(f)


# In[102]:


xgb_loaded


# In[ ]:





# In[122]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask_ngrok import run_with_ngrok 


# In[124]:


# In[2]
app = Flask(__name__)
#run_with_ngrok(app)
filename = 'C:/Users/HARSHA/Documents/Deployment/final_trained_model.pkl'
model = pickle.load(open(filename, 'rb'))


# In[125]:


def get_input(inp):
    s = ""
    for w in inp:
        s = s + " " + w
    #print(s)
    s_ = preprocess(s) 
    s = ""
    for char in s_:
        s = char + " " + s
    df = pd.DataFrame(data = {'comment_text' : s},index=[0])
    X = cv.transform(df['comment_text'])
    return X


# In[126]:


@app.route('/',endpoint='home') 
def home():
    return render_template('index.html')


# In[127]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    inp_text = [str(x) for x in request.form.values()]
    X=get_input(inp_text)
    outputs = loaded_model.predict_proba(X)
    d={}
    d['toxic']=outputs[0][0]
    d['severe toxic']=outputs[0][1]
    d['obscene']=outputs[0][2]
    d['threat']=outputs[0][3]
    d['insult']=outputs[0][4]
    d['identity hate']=outputs[0][5]
    #dc={}
    #dc['toxic']=preds[0][0]
    #dc['severe toxic']=preds[0][1]
    #dc['obscene']=preds[0][2]
    #dc['threat']=preds[0][3]
    #dc['insult']=preds[0][4]
    #dc['identity hate']=preds[0][5]


    return render_template('index.html', prediction_probabilities='Prediction probabilities are {}'.format(d))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




