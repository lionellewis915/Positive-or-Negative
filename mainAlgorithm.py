'''
Created by: Lionel Lewis with help from Gareth Dwyer

Simple Machine Learning algorithm that determines whether a statement is positive or negative

This algorithm uses the SCI-LEARN package and runs in python 3.
'''

from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

positive_texts = [
	"we love you",
	"they love us",
	"he is good",
	"They love mary"
]

negative_texts = [
	"we hate you",
	"they hate us",
	"you are bad",
	"he is bad",
	"we hate mary"
]

test_texts = [
	"they love mary",
	"they are good",
	"why do you hate mary",
	"they are almost always good",
	"we are very bad",
	"we enjoy being with you",
	"i love you"
]

training_texts = negative_texts + positive_texts
training_labels = ["negative"] * len(negative_texts) + ["positive"] * len(positive_texts)

vectorizer = CountVectorizer()
vectorizer.fit(training_texts)
print(vectorizer.vocabulary_)
print()

training_vectors = vectorizer.transform(training_texts)
testing_vectors = vectorizer.transform(test_texts)



classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)
predictions = classifier.predict(testing_vectors)
print(predictions)
print()

tree.export_graphviz(
	classifier,
	out_file = 'tree.dot',
	feature_names = vectorizer.get_feature_names(),
)

def manual_classify(text):
	if "hate" in text:
		return "negative"

	if "bad" in text:
		return "negative"

	return "positive"

predictions = []
for text in test_texts:
	prediction = manual_classify(text)
	predictions.append(prediction)
print(predictions)
print()
