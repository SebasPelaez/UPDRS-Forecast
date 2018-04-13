import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def Regression(X,Y,groups,grade):
    lpgo = GroupKFold(n_splits=14)
    MAE_fo = []
    MAE_so = []
    ECM_fo = []
    ECM_so = []
    
    #Se añaden los grados del polinomio a las caracteristicas
    poly = PolynomialFeatures(degree=grade)
    X = poly.fit_transform(X)
    
    for train_index, test_index in lpgo.split(X, Y, groups):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = Y[train_index], Y[test_index]
      
      y_train_fo = y_train[:,0]
      y_train_so = y_train[:,1]
      y_test_fo = y_test[:,0]
      y_test_so = y_test[:,1]
      
      #Normalización de los datos
      
      sc_X = StandardScaler()
      X_train = sc_X.fit_transform(X_train)
      X_test = sc_X.transform(X_test)
      
      # Ajustar el modelo a la regresión simple
      fo = LinearRegression()
      so = LinearRegression()
      
      fo.fit(X_train, y_train_fo)
      so.fit(X_train, y_train_so)
      
      # Predecir los resultados de prueba
      y_pred_fo = fo.predict(X_test)
      y_pred_so = so.predict(X_test)
    
      ECM_fo.append(mean_squared_error(y_test_fo,y_pred_fo))
      ECM_so.append(mean_squared_error(y_test_so,y_pred_so))
      
      MAE_fo.append(mean_absolute_error(y_test_fo,y_pred_fo))
      MAE_so.append(mean_absolute_error(y_test_so,y_pred_so))
      
    print("El error cuadratrico medio de validación para la primer salida es (ECM):", np.mean(ECM_fo))
    print("El error cuadratrico medio de validación para la segunda salida es (ECM):", np.mean(ECM_so))
    
    print("El porcentaje de error medio absoluto de validación para la primer salida es (MAE):", np.mean(MAE_fo))
    print("El porcentaje de error medio absoluto de validación para la segunda salida es (MAE):", np.mean(MAE_so))