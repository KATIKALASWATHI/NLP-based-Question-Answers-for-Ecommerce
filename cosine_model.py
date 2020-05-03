import pandas as pd 
import numpy as np
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_percentron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import os

os.getcwd()
os.chdir('C:\\Users\\Swathi\\Desktop\\NLP-project')
df=pd.read_csv('file:///C:/Users/Swathi/Desktop/myproject/qa_Electronicdata.csv')
df_new=df.drop(['questionType', 'asin', 'answerTime', 'unixTime', 'answerType'], axis=1)

df=pd.read_csv('file:///C:/Users/Swathi/Desktop/NLP-project/processed_data.csv')

##df_new.head()
df_final=df.drop_duplicates(subset='keywords')
df_final.shape
df_final.dropna(inplace=True)
df_final.isnull().sum()
df_final.shape

stopwords_list=stopwords.words('english')

lemmatizer=WordNetLemmatizer()
def my_tokenizer(doc):
    words=word_tokenize(doc)
    pos_tags=pos_tag(words)
    
    non_stopwords=[w for w in pos_tags if not w[0].lower() in stopwords_list]
    
    non_punctuation=[w for w in non_stopwords if not w[0] in string.punctuation]
    lemmas=[]
    
    for w in non_punctuation:
        if w[1].startswith('J'):
            pos=wordnet.ADJ
        elif w[1].startswith('v'):
            pos=wordnet.VERB
        elif w[1].startswith('N'):
            pos=wordnet.NOUN
        elif w[1].startswith('R'):
            pos=wordnet.ADV
        else:
            pos=wordnet.NOUN
            
        lemmas.append(lemmatizer.lemmatize(w[0], pos))
    return lemmas
    
tfidf_vectorizer=TfidfVectorizer(tokenizer= my_tokenizer)
tfidf_matrix=tfidf=tfidf_vectorizer.fit_transform(tuple(df_final['keywords']))

pickle.dump(tfidf_vectorizer, open('transform.pkl', 'wb'))

#print(tfidf_matrix.shape)
def ask_question(question):
    query_vect=tfidf_vectorizer.transform([question])
    similarity=cosine_similarity(query_vect, tfidf_matrix)
    
    top_5_simmi=similarity[0].argsort()[-5:][::-1]
    
    count=1
    for i in top_5_simmi:
        print('Question:-', count,':', df_final.iloc[i]['question'])
        print('/n')
        print('answer:', df_final.iloc[i]['answer'])
        print('/n')
        print('Accuracy is:{:,.2%}', format(similarity[0,i]))
        count+=1
        
 filename='nlp_model.pkl'
 pickle.dump(filename, open(filename, 'wb'))
ask_question(input('Hello, please enter the keywords for /n a question you want to search for: '))
            
            
            
            
            
            
            
            
            
            
            