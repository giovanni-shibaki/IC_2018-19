###  EXEMPLO DE PROGRAMA PARA RANDOMTREE CLISSIFIER  ###################################################################



import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

features = pd.read_csv('CaracteristicaT.csv')
features.head(5)

y = np.array(features['STATUS'])
X = features.drop('STATUS', axis=1)

# Divide o DataSet, uma parte para treino e outra para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% para Treino e 30% para Teste


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=200)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,'%')


# PODE-SE APÓS O TREINO TENTAR PREVER QUAL É O TIPO COM 86% DE CHANCE DE ACERTO
dados = [[7,12,25,39,0,0,0,0,1,1,1,2,7,14,20,27]]
previsao = clf.predict(dados)
prob = clf.predict_proba(dados)
print('Status previsto: ',previsao)
print('Probabilidade (0 , 1): ',prob)

#PARA SALVAR
    #import pickle
    #with open('net.pkl', 'wb') as f:
    #pickle.dump(clf,f)
#PARA SALVAR

#PARA LER
    #with open('net.pkl', 'wb') as f:
    #net = pickle.load(f)
#PARA LER

import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=list(features.drop('STATUS', axis=1))).sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Importância das Caracteristicas na previsão')
plt.ylabel('Caracteristicas')
plt.title("Características Importantes")
plt.legend()
plt.show()
