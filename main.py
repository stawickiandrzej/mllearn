import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier

wine = load_wine()

features, target = wine.data, wine.target

dummy_clf = DummyClassifier(strategy="most_frequent")

dummy_clf.fit(features, target)

dummy_clf.predict(features)

print(dummy_clf.score(features, target))
