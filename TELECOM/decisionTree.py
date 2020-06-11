
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

features = pd.read_csv('CaracteristicaT.csv')
features.head(5)

y = np.array(features['STATUS'])
X = features.drop('STATUS', axis=1)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


#Import Random Forest Model
from sklearn.tree import DecisionTreeClassifier

#Create a Gaussian Classifier
clf=DecisionTreeClassifier()

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# PODE-SE APÓS O TREINO TENTAR PREVER QUAL É O TIPO COM 86% DE CHANCE DE ACERTO
dados = [[12,21,21,21,0,2,2,2,0,0,0,0,0,0,0,0]]
previsao = clf.predict(dados)
prob = clf.predict_proba(dados)
print('Status previsto: ',previsao)
print('Probabilidade (0 , 1): ',prob)



