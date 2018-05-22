# Importing las bibliotecas
import numpy as np
import pandas as pd

from Regression.MultipleLinearRegression import Regression
from Forest.RandomForest import RandomForest
from SVR.SupportVectorRegression import SupportVectorRegression
from NeuronalNet.NeuronalNetwork import NeuronalNetwork
from VentanaDeParzen.VentanaParzen import VentanaParzen

from ReduccionDeDimensiones.AnalisisCaracteristicas import CoeficientePearson
from ReduccionDeDimensiones.SeleccionCaracteristicas import SFS

#Organizar las variables de entrada y las variables a predecir
#full_data = pd.read_csv('https://raw.githubusercontent.com/SebasPelaez/ProyectoSimulacion/master/Data.csv?token=AOThxXzOdjEisj_b_dDeBvb7_oIZRPK1ks5a1hyAwA%3D%3D')
full_data = pd.read_csv('./Data.csv')
groups = full_data.iloc[:, 0:1].values
x1 = full_data.iloc[:, 1:4].values
x2 = full_data.iloc[:, 6:22].values
X = np.concatenate((x1,x2),axis=1)
Y = full_data.iloc[:, 4:6].values

grade = int(input("Que grado del Polinomio: "))
Regression(X,Y,groups,grade)

n_tress = int(input("Número de árboles: "))
RandomForest(X,Y,groups,n_tress)

opcion = int(input("Tipo de Kernel (1) Para Linear (2) Para RBF: "))
if opcion == 1:
    kernel = 'linear'
else:
    kernel = 'rbf'
SupportVectorRegression(X,Y,groups,'rbf')

print("Red Neuronal Artificial")
NeuronalNetwork(X,Y,groups)

opcion = float(input("Ingrese el ancho de la ventana"))
VentanaParzen(X,Y,groups,opcion)

CoeficientePearson(X,Y)

selected_features = SFS(X,Y[:,0])

X_sfs = X[:,(0,2,5,18)]
opcion = float(input("Ingrese el ancho de la ventana"))
VentanaParzen(X_sfs,Y,groups,opcion)