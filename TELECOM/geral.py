### IMPORTS PARA CONVERSÃO DE DADOS, LEITURA DO CSV E GERAÇÃO DE GRÁFICOS  #############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### IMPORTS PARA CONVERSÃO DE DADOS, LEITURA DO CSV E GERAÇÃO DE GRÁFICOS  #############################################






### IMPORT OPICIONAL PARA MOSTRAR AVISOS  ##############################################################################

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

### IMPORT OPICIONAL PARA MOSTRAR AVISOS  ##############################################################################







#Exemplo de dataset já pronto: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html






###  IMPORTAR O CSV COM OS DADOS NECESSÁRIOS  ##########################################################################

features = pd.read_csv('CaracteristicaT.csv') #leitura das caracteristicas

###  IMPORTAR O CSV COM OS DADOS NECESSÁRIOS  ##########################################################################






###  SEPARA O DATASET, Y É O TARGET, OU SEJA, O QUE ELE PRETENDE PREVER E X SÃO AS CARACTERISTICAS ANALISADAS PARA CHEGAR NO TARGET  ###

y = np.array(features['STATUS']) #leitura dos status
X = features.drop('STATUS', axis=1) #remove a coluna status

###  SEPARA O DATASET, Y É O TARGET, OU SEJA, O QUE ELE PRETENDE PREVER E X SÃO AS CARACTERISTICAS ANALISADAS PARA CHEGAR NO TARGET  ###







###  Divide o dataset, X% para treino e Y% para teste  #################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

###  Divide o dataset, X% para treino e Y% para teste  #################################################################







###  MÉTODO DO SCKIT LEARN PARA REGISTRAR ACCURACY E LOSS  #############################################################

from sklearn.metrics import accuracy_score, log_loss

###  MÉTODO DO SCKIT LEARN PARA REGISTRAR ACCURACY E LOSS  #############################################################








###  Importar Métodos de IA  ###########################################################################################

from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

###  Importar Métodos de IA  ###########################################################################################








###  LISTA DE CLASSIFICADORES QUE SERÃO UTILIZADOS  ####################################################################

classifiers = [
    KNeighborsClassifier(3), #knn
    #SVC(kernel="rbf", C=0.025, probability=True),
    #NuSVC(probability=True),
    DecisionTreeClassifier(), #Árvore de Decisão
    MLPClassifier(alpha=1), #rede neural
    RandomForestClassifier(), #Random Forest
    AdaBoostClassifier(),
    GradientBoostingClassifier(), #Gradient Boosting
    GaussianNB(), #Naive Bayes
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression() #Regressão Logística
]

###  LISTA DE CLASSIFICADORES QUE SERÃO UTILIZADOS  ####################################################################








### PARA GERAR GRÁFICOS MAIS TARDE  ####################################################################################
log_cols = ["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
### PARA GERAR GRÁFICOS MAIS TARDE  ####################################################################################








### NO FOR ABAIXO TODOS OS CLASSIFICADORES SELECIONADOS ACIMA SERÃO TESTADOS, TREINADOS E FARÃO UMA PREVISÃO TESTE  ####

for clf in classifiers:
    clf.fit(X_train, y_train) #Executando o treino (treino e teste)
    name = clf.__class__.__name__

    print("=" * 60)
    print(name)

    print('****Resultados****')
    #accuracy
    train_predictions = clf.predict(X_test) #Executando teste
    acc = accuracy_score(y_test, train_predictions) # Verifica o accuracy
    print("Probabilidade de Acerto: {:.4%}".format(acc))

    #loss
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))

    # Teste de um único dado
    dados = [[0,16,23,23,0,0,0,0,6,10,10,10,4,11,16,16]] #Tem status 0
    previsao = clf.predict(dados)
    prob = clf.predict_proba(dados)
    print('Status a ser previsto: 0 - Cancelado')
    print('Status previsto: ', previsao)
    print('Probabilidade (0 , 1): ', prob)


    # Objetos para a criação dos gráficos
    log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
    log = log.append(log_entry)

print("=" * 90)





###  Gráfico de ACCURACY  ##############################################################################################

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.title('Probabilidade de Acerto por Classificador')
plt.show()

###  Gráfico de ACCURACY  ##############################################################################################







###  Gráfico de LOSS  ##################################################################################################

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()

###  Gráfico de LOSS  ##################################################################################################








###  PARA SALVAR O CLASSIFICADOR  ######################################################################################

    #import pickle
    #with open('net.pkl', 'wb') as f:
    #pickle.dump(clf,f)

###  PARA SALVAR O CLASSIFICADOR  ######################################################################################




###  PARA LER O CLASSIFICADOR  #########################################################################################

    #import pickle
    #with open('net.pkl', 'wb') as f:
    #net = pickle.load(f)

### PARA LER O CLASSIFICADOR   #########################################################################################
