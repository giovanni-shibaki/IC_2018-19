import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import pickle
with open('net.pkl', 'rb') as f:
    net = pickle.load(f)

# PODE-SE APÓS O TREINO TENTAR PREVER QUAL É O TIPO COM 86% DE CHANCE DE ACERTO
dados = [[12,21,21,21,0,2,2,2,0,0,0,0,0,0,0,0]]
previsao = net.predict(dados)
prob = net.predict_proba(dados)
print('Status previsto: ',previsao)
print('Probabilidade (0 , 1): ',prob)
