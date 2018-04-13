# Importing las bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from ProyectoSimulacion.MultipleLinearRegression import Regression

#Organizar las variables de entrada y las variables a predecir
full_data = pd.read_csv('https://raw.githubusercontent.com/SebasPelaez/ProyectoSimulacion/master/Data.csv?token=AOThxXzOdjEisj_b_dDeBvb7_oIZRPK1ks5a1hyAwA%3D%3D')
groups = full_data.iloc[:, 0:1].values
x1 = full_data.iloc[:, 1:4].values
x2 = full_data.iloc[:, 6:22].values
X = np.concatenate((x1,x2),axis=1)
y = full_data.iloc[:, 4:6].values

lpgo = GroupKFold(n_splits=14)
MAE_fo = []
MAE_so = []
ECM_fo = []
ECM_so = []

poly = PolynomialFeatures(degree=1)
X = poly.fit_transform(X)

for train_index, test_index in lpgo.split(X, y, groups):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
   
  # Ajustar el modelo a la regresión simple
  y_test, y_pred = Regression(X_train,X_test,y_train,y_test)
  
  ECM_fo.append(mean_squared_error(y_test[:,0],y_pred[:,0]))
  ECM_so.append(mean_squared_error(y_test[:,1],y_pred[:,1]))
  
  MAE_fo.append(mean_absolute_error(y_test[:,0],y_pred[:,0]))
  MAE_so.append(mean_absolute_error(y_test[:,1],y_pred[:,1]))
  
print("El error cuadratrico medio de validación para la primer salida es (ECM):", np.mean(ECM_fo))
print("El error cuadratrico medio de validación para la segunda salida es (ECM):", np.mean(ECM_so))

print("El porcentaje de error medio absoluto de validación para la primer salida es (MAE):", np.mean(MAE_fo))
print("El porcentaje de error medio absoluto de validación para la segunda salida es (MAE):", np.mean(MAE_so))
