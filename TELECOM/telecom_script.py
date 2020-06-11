
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
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

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

#Importância das cacrateristicas
import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=list(features.drop('STATUS', axis=1))).sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Importância do Atributo')
plt.ylabel('Atributos')
plt.title("Gráfico da importância dos atributos")
plt.legend()
plt.show()



feature_list = list(features.drop('STATUS', axis=1))

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = clf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png')

