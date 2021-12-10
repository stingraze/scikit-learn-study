#Based on code from: https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
#This code visualizes text and URL pair as a vector with matplotlib.
#Original code by Jason Brownlee, code edited by Tsubasa Kato (@_stingraze on Twitter) on December 10th, 2021
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pylab as plt

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

#Visualize 1st vector and second vector
plt.spy(vector)
plt.figure(0)
plt.spy(vector2)
