# Importing necessary packages
# In[1]

import pandas as pd
import numpy as np
import seaborn as sns 

# In[2]

train = pd.read_csv(r"F:\Jigsaw_toxic_datasets\train.csv")
test = pd.read_csv(r"F:\Jigsaw_toxic_datasets\test.csv")

# In[3]

train 

# In[4]

test 

# In[5]

train.dtypes 

# In[6]

train.describe() 

# In[7] 

# We will know that how many rows and columns were there in the dataset in this particular cell by using "count"

rows_count, columns_count = train.shape
print('Total Number of rows :', rows_count)
print('Total Number of columns :', columns_count)

# In[8]

train.info() 

# In[9] 

train.isnull() 

# In[10] 

# we can see there is no missing value in the dataframe.
# i.e False refers that the dataset does not contain the null values. You can see clearly in below section

train.isnull().sum() 

# In[11] 

# Here the duplicated data is assigned to the variable "dup" and it shows whether the data contains duplicated data or not.

dup = train.duplicated() 

# In[12] 

dup 

# In[13]

# Now we drop some unnecessary columns from the dataset

test = test.drop(['id'], axis=1 ) 

# In[14]

test.head() 

# In[15] 

train.nunique() 

# In[16] 

test.nunique()  

# In[17] 

print(train.shape, test.shape)

# In[18]

train["toxic"].value_counts().to_frame() 

# In[19] 

train["severe_toxic"].value_counts().to_frame() 

# In[20] 

train["obscene"].value_counts().to_frame()

# In[21] 

train["threat"].value_counts().to_frame() 

# In[22]

train["insult"].value_counts().to_frame()

# In[23]

train["identity_hate"].value_counts().to_frame()

# In[24]

train.corr()

# In[25]

# Heatmap
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))
sns.heatmap(train.corr(), annot = True)

# In[26]

import matplotlib.pyplot as plt

sns.countplot(x='toxic',data=train)
plt.title('Distribution of Toxic Comments')

# In[27]

# Pairplot

sns.pairplot(train.iloc[:,1:])
plt.show() 

# In[28] 

# Data Preprocessing

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import warnings
warnings.filterwarnings('ignore')

# In[29] 

stop_words = set(stopwords.words('english')) 

# In[30]

import contractions 
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

# In[31]
    
for i in range(0,len(train)):
    char = []
    s = ""
    char = preprocess(train['comment_text'][i])  
    for w in char:
        s = s + w + " "
    train['comment_text'][i] = s 
    
# In[32] 
    
train 

# In[33]

for i in range(0,len(test)):
    char = []
    s = ""
    char = preprocess(test['comment_text'][i])  
    for w in char:
        s = s + w + " "
    test['comment_text'][i] = s 

# In[34]
    
test 

# In[35]

# Model Training 

from sklearn.feature_extraction.text import CountVectorizer

X = train['comment_text']
Y = train.drop(['id','comment_text'],axis=1) 

# In[36]

from sklearn.model_selection import train_test_split

Xtr,Xts,Ytr,Yts = train_test_split(X,Y,test_size = 0.2) 
print(Xtr.shape,Xts.shape,Ytr.shape,Yts.shape) 

# In[37] 

cv = CountVectorizer()
Xtr = cv.fit_transform(Xtr)
Xts = cv.transform(Xts)

# In[] 
import pickle 

pickle.dump(cv, open('C:/Users/HARSHA/Documents/Jigsaw_Comment_classification/tranform.pkl', 'wb'))
 
# In[]


# In[38]

from sklearn import metrics
import itertools 

# In[39] 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score 
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# In[40]

model_log = OneVsRestClassifier(LogisticRegression()) 
model_log.fit(Xtr,Ytr)

# In[41]

y_pred = model_log.predict(Xts)
y_pred 

# In[42] 

score = metrics.accuracy_score(Yts,y_pred)
print("accuracy :  %0.3f" % score) 

# In[43]

from xgboost import XGBClassifier

# In[44]

xgb = OneVsRestClassifier(XGBClassifier())
xgb.fit(Xtr, Ytr) 

# In[45]

x_pred = xgb.predict(Xts)
x_pred

# In[46]

score = metrics.accuracy_score(Yts,x_pred)
print("accuracy :  %0.3f" % score) 

# In[47]

f_pred = xgb.predict_proba(np.round(Xts,2)) 

# In[48] 

f_pred 

# In[49]

new = pd.DataFrame(f_pred) 

# In[50]

final_file = new.to_csv('final.csv') 

# In[51]

new.rename(columns = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}, inplace = True)

# In[52]

new 


# As 91.7 is the best accuracy score, so XGBoost is the best algorithm.

# In[53]

import pickle

# In[54]

with open('C:/Users/HARSHA/Documents/Jigsaw_Comment_classification/trained_model.pkl', 'wb') as f:
    pickle.dump(xgb, f) 

# In[55]
    
with open('C:/Users/HARSHA/Documents/Jigsaw_Comment_classification/trained_model.pkl', 'rb') as f:
    xgb_loaded = pickle.load(f) 

# In[56]
    
xgb_loaded 
 
# In[59]



