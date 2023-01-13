from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
from sklearn import tree
import sys
import matplotlib.pyplot as plt
import matplotlib
from dataExtract import read_csv
import csv

matplotlib.use('Agg')
path = Path('.')

f = open('pred.csv', 'w', newline='')
csvwriter = csv.writer(f)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
Y = ['Survived']
train, test = read_csv(path, 'scikit')
dtree = DecisionTreeClassifier()
dtree.fit(train[features], train[Y])
tree.plot_tree(dtree, feature_names = features)
plt.savefig(fname = "dtree.png")
predictions = dtree.predict(test[features])
csvwriter.writerow(["PassengerId","Survived"])
m = lambda z: [csvwriter.writerow([str(i + 892), str(z[i])]) for i in range(len(z))]
m(predictions)