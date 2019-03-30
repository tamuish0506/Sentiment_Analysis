print()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
max_features=1000

from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

google_review=pd.read_csv('desktop/googleplaystore_user_reviews.csv')
google_review.head()
google_review.columns
google_review.columns=[x.lower() for x in google_review.columns]


sentiment_data=pd.concat([google_review.translated_review, google_review.sentiment], axis=1)
sentiment_data.isnull().sum()
sentiment_data.dropna(axis=0, inplace=True)
sentiment_data.head(10)
sentiment_data.shape

sentiment_data.sentiment.unique()
sentiment_data['sentiment']=sentiment_data['sentiment'].map({'Positive':0, 
              'Negative':1, 'Neutral':2})

sentiment_data.sentiment.value_counts()

sentiment_data.head()
sns.countplot(sentiment_data.sentiment)
plt.title('Counts of Sentiments')
plt.show()

#drop symbols and lowercase each word
#draw one sample to test then apply on each review
first_sample=sentiment_data.translated_review[9]
first_sample
sample = re.sub("[^a-zA-Z]"," ",first_sample)#remove punctuations
sample
sample = sample.lower()
print("[{}] convert to \n[{}]".format(first_sample,sample))

#process stop words
#split a sentence by word (tokens)
sample=nltk.word_tokenize(sample)#tokenize
print(sample)
#delet unnecessary words, such as like, it, I
sample = [word for word in sample if not word in set(stopwords.words("english"))]
print(sample)

#Lemmatization: convert words to stem
lemma=nltk.WordNetLemmatizer()  
sample=[lemma.lemmatize(word) for word in sample]
sample=" ".join(sample)
sample
#so far removal of symbols and stopwords, lowercase, tokenization and lammatization
#are already completed in one sample, now we have to apply these functions to
#all reviews!!

review_list=[]
for i in sentiment_data.translated_review:
    review=re.sub("[^a-zA-Z]"," ",i)
    review=review.lower()
    review=nltk.word_tokenize(review)
    review=[word for word in review if not word in set(stopwords.words('english'))]
    lemma=nltk.WordNetLemmatizer()
    review=[lemma.lemmatize(word) for word in review]
    review=" ".join(review)
    review_list.append(review)
    
review_list[:5]

#We have bag words now and seek for clean and relevant words.
count_of_vectors=CountVectorizer(max_features=max_features)
sparce_matrix=count_of_vectors.fit_transform(review_list).toarray()
all_words=count_of_vectors.get_feature_names()
print('50 The Most Common Words: ', all_words[0:50])

#Find a model that will assign words to the corresponding sentiment ->
#prediction!!!

#Split the dataset
y = sentiment_data.iloc[:,1].values
x= sparce_matrix

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,
                                                    random_state=1)

#First model: Gaussian Naive Bayse Classifier
nb=GaussianNB()
model_nb=nb.fit(x_train, y_train)
acc_nb=nb.score(x_test, y_test)
print('Accuracy for Naive Bayse: ', round(acc_nb, 3))
#Accuracy is around 0.58...quite low. Let's find out if other model performs better.

#Confusion Matrix
nb_pred=nb.predict(x_test)
cm_nb=confusion_matrix(y_test, nb_pred)
print(cm_nb)
print(classification_report(y_test, nb_pred))
#Plot confusion matrix
name=['Positive', 'Negative', 'Neutral']
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm_nb,annot=True,linewidth=5,fmt=".0f",ax=ax)
plt.xlabel('NB_Prediction')
plt.ylabel('Real Value')
ax.set_xticklabels(name)
ax.set_yticklabels(name)
plt.show()

#Second model: Random Forest Classifier
rf=RandomForestClassifier(n_estimators=10,random_state=42)
model_rf=rf.fit(x_train,y_train)
acc_rf=rf.score(x_test,y_test)
print('Accuracy for Random Forest: ', round(acc_rf, 3))

#Confusion Matrix
rf_pred=rf.predict(x_test)
cm_rf=confusion_matrix(y_test, rf_pred)
print(cm_rf)
print(classification_report(y_test, rf_pred))
#Plot confusion matrix
name=['Positive', 'Negative', 'Neutral']
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm_rf,annot=True,linewidth=5,fmt=".0f",ax=ax)
plt.xlabel('RF_Prediction')
plt.ylabel('Real Value')
ax.set_xticklabels(name)
ax.set_yticklabels(name)
plt.show()

#Third model: Logistic Regression
lr=LogisticRegression()
model_lr=lr.fit(x_train,y_train)
acc_lr=lr.score(x_test,y_test)
print('Accuracy for Logistic Regression: ', round(acc_lr, 3))

#Confusion Matrix
lr_pred=lr.predict(x_test)
cm_lr=confusion_matrix(y_test, lr_pred)
print(cm_lr)
print(classification_report(y_test, lr_pred))
#Plot confusion matrix
name=['Positive', 'Negative', 'Neutral']
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm_lr,annot=True,linewidth=5,fmt=".0f",ax=ax)
plt.xlabel('LR_Prediction')
plt.ylabel('Real Value')
ax.set_xticklabels(name)
ax.set_yticklabels(name)
plt.show()

