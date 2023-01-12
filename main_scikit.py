from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
from sklearn import tree
import sys
import matplotlib.pyplot as plt
import matplotlib
from dataExtract import read_csv
 

matplotlib.use('Agg')
path = Path('.')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
Y = ['Survived']
train, test = read_csv(path, 'scikit')
dtree = DecisionTreeClassifier()
dtree.fit(train[features], train[Y])
tree.plot_tree(dtree, feature_names = features)
plt.savefig(fname = "dtree.png")
predictions = tree.predict(test[features])
