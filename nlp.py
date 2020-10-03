import pandas as pd
import nltk
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk.tokenize import word_tokenize

loc = ("C:\\Users\\Yash\\Desktop\\NLP\\NLP.xlsx") 

df = pd.read_excel(loc)

df = df[df.columns[5:10]]
q1 = df[df.columns[0]].tolist()
q2 = df[df.columns[1]].tolist()
q3 = df[df.columns[2]].tolist()
q4 = df[df.columns[3]].tolist()
q5 = df[df.columns[4]].tolist() 
q1=q1[1:]

train = [('The study material is not satisfactory', 'neg'),
         ('It is difficult to search the required information', 'neg'),
         ('The study material is easily available', 'pos'),
         ('Any new information can be accessed with comparative ease', 'pos'),
         ('Yes', 'pos'),
         ('Not sufficient or precise materials', 'neg'),
         ('Internet and author\'s point varies', 'neg'),
         ('There were reference books available to refer for the course','pos'),
         ('No', 'neg'),
         ('The reference books for NLP are very good','pos'),
         ('The study materical is not satisfactory','neg'),
         ('I am satisfied', 'pos'),
         ('There is enough material available', 'pos')]

all_words = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]
classifier = nltk.NaiveBayesClassifier.train(t)
test_sentence = "not satisfied"

test_sent_features = {word: (word in word_tokenize(test_sentence.lower())) for word in all_words}

print(classifier.classify(test_sent_features))
