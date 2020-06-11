import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

features = pd.read_csv('CaracteristicaT.csv')

y = np.array(features['STATUS'])
X = features.drop('STATUS', axis=1)

# Split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear',cache_size=1000)
svclassifier.fit(X_train, y_train)


#y_pred = svclassifier.predict(X_test)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
 #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dados = [[12,21,21,21,0,2,2,2,0,0,0,0,0,0,0,0]]
y_pred = svclassifier.predict(dados)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))