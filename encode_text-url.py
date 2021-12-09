#Based on code from: https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
#Original code by Jason Brownlee, code edited by Tsubasa Kato (@_stingraze on Twitter) on December 10th, 2021
#Tested to work on Google Colaboratory
from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
url = ["https://www.google.com"]
text2 = ["The red brown fox jumped over the fast dog."]
url2 = ["https://www.superai.online"]
# create the transform
vectorizer = CountVectorizer()
vectorizer2 = CountVectorizer()
# tokenize and build vocab
text = text  + list(url)
text2 = text2 + list(url2)
vectorizer.fit(text)
vectorizer2.fit(text2)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer2.vocabulary_)
# encode document
vector = vectorizer.transform(text)
vector2 = vectorizer.transform(text2)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())
# summarize encoded vector2
print(vector2.shape)
print(type(vector2))
print(vector2.toarray())
