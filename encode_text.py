#Code mostly from https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
#Edited to make a new sample (added vectorizer2, vector2 etc.)
#Edited by Tsubasa Kato, original code by Jason Brownlee
from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
text2 = ["The red brown fox jumped over the fast dog."]
# create the transform
vectorizer = CountVectorizer()
vectorizer2 = CountVectorizer()
# tokenize and build vocab
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
