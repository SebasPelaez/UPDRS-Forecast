# Importing las bibliotecas
import numpy as np
import pandas as pd

from MultipleLinearRegression import Regression
from RandomForest import RandomForest
from SupportVectorRegression import SupportVectorRegression
from NeuronalNetwork import NeuronalNetwork

#Organizar las variables de entrada y las variables a predecir
full_data = pd.read_csv('https://raw.githubusercontent.com/SebasPelaez/ProyectoSimulacion/master/Data.csv?token=AOThxXzOdjEisj_b_dDeBvb7_oIZRPK1ks5a1hyAwA%3D%3D')
groups = full_data.iloc[:, 0:1].values
x1 = full_data.iloc[:, 1:4].values
x2 = full_data.iloc[:, 6:22].values
X = np.concatenate((x1,x2),axis=1)
Y = full_data.iloc[:, 4:6].values

opcion = int(input("Que modelo desea ejecutar: "))

#grade = int(input("Que grado del Polinomio: "))
#Regression(X,Y,groups,grade)

#n_tress = int(input("Número de árboles: "))
#RandomForest(X,Y,groups,n_tress)

#kernel = int(input("Tipo de Kernel (1) Para Linear (2) Para RBF: "))
#SupportVectorRegression(X,Y,groups,kernel)

print("Red Neuronal Artificial")
NeuronalNetwork(X,Y,groups)