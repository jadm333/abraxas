#%%
import pandas as pd
import spacy
from gensim.models import Phrases
from funcs import entidades, quitarpuntuacion, bigramas
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
#%%

train = pd.read_csv('data/train.csv',index_col=0)
test = pd.read_csv('data/test.csv')
subm = pd.read_csv('data/sample_submission.csv')

#%% Crear una columna de limpio
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['limpio'] = 1-train[label_cols].max(axis=1)


#%% BORRAR NA's

train['comment_text'].fillna("unknown", inplace=True)
test['comment_text'].fillna("unknown", inplace=True)

#%% Preprocesamiento
nlp = spacy.load('en')
train = quitarpuntuacion(train)
docs = train.comment_text

#%%
%%time
docs = entidades(docs,nlp)
docs = bigramas(docs)

#%%
pickle.dump( docs, open( "save.p", "wb" ) )

#%%

docs = pickle.load( open( "save.p", "rb" ) )

#%%
doc_vacios = []
for i in range(len(docs)):
    if len(docs[i]) == 0:
        doc_vacios.append(i)
#%% Analisis descriptivo
# TODO: Analisis descriptivo

#%% Modelo
##Regreasamos a string

train['texto_limpio'] = docs
#train['texto_limpio'] = train['texto_limpio'].apply(lambda x: ' '.join(x))
del train['comment_text']
del train['limpio']
#%%Dividir

X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,6], train.iloc[:,0:6],test_size=0.2)

#%%
vectorizer = TfidfVectorizer(stop_words = 'english',\
                             vocabulary=None,\
                             analyzer='word',\
                             lowercase=True,\
                             ngram_range=(1, 1),\
                             max_df=1.0,\
                             min_df=1)

tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)
#%% Logistic Regression


reglog = LogisticRegression()
presicion_reglog = dict()

for label in label_cols:
    y = y_train[label]
    reglog.fit(tfidf_train, y)
    y_hat = reglog.predict(tfidf_test)
    presicion_reglog[label] = accuracy_score(y_test[label], y_hat)
#%%
## Naive Bayes
    
naive_bayes = MultinomialNB()
presicion_naive_bayes = dict()

for label in label_cols:
    y = y_train[label]
    naive_bayes.fit(tfidf_train, y)
    y_hat = naive_bayes.predict(tfidf_test)
    presicion_naive_bayes[label] = accuracy_score(y_test[label], y_hat)
#%% SGDClassifier

SGDC = SGDClassifier()
presicion_SGDC = dict()

for label in label_cols:
    y = y_train[label]
    SGDC.fit(tfidf_train, y)
    y_hat = SGDC.predict(tfidf_test)
    presicion_SGDC[label] = accuracy_score(y_test[label], y_hat)

#%% Mejor clasificador

print(sum(list(presicion_reglog.values()))/6)
print(sum(list(presicion_naive_bayes.values()))/6)
print(sum(list(presicion_SGDC.values()))/6)

#%%submission
tfidf_train_c =vectorizer.fit_transform(train['texto_limpio'])
tfidf_sub = vectorizer.transform(test['comment_text'])

for label in label_cols:
    y = train[label]
    reglog.fit(tfidf_train_c, y)
    test[label] = reglog.predict_proba(tfidf_sub)[:,1]
#%%

#%%
subm = test
del subm["comment_text"]
#%%
subm.to_csv('subs/submission4.csv',index=False)
