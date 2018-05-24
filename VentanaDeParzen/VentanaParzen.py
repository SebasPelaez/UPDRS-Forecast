import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

from VentanaDeParzen.KernelRegression import KernelRegression

def VentanaParzen(X,Y,groups,gamma):
    lpgo = GroupKFold(n_splits=14)
    MAE_fo = []
    MAE_so = []
    ECM_fo = []
    ECM_so = []
    MAPE_fo = []
    MAPE_so = []
    R2_SCORE_fo = []
    R2_SCORE_so = []
    kr = KernelRegression(kernel="rbf", gamma=gamma)
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
      
      # Ajustar el modelo de ventana de parzen
      M = len(X_test)
      y_pred_fo = np.zeros((M,1))
      y_pred_so = np.zeros((M,1))
      
      y_pred_fo = kr.fit(X_train, y_train_fo).predict(X_test)
      y_pred_so = kr.fit(X_train, y_train_so).predict(X_test)
    
      ECM_fo.append(mean_squared_error(y_test_fo,y_pred_fo))
      ECM_so.append(mean_squared_error(y_test_so,y_pred_so))
    
      MAE_fo.append(mean_absolute_error(y_test_fo,y_pred_fo))
      MAE_so.append(mean_absolute_error(y_test_so,y_pred_so))
    
      MAPE_fo.append(np.mean(np.abs((y_test_fo - y_pred_fo) / y_test_fo)) * 100)
      MAPE_so.append(np.mean(np.abs((y_test_so - y_pred_so) / y_test_so)) * 100)
      
      R2_SCORE_fo.append(r2_score(y_test_fo, y_pred_fo))
      R2_SCORE_so.append(r2_score(y_test_so, y_pred_so))
    
    print("El error cuadratrico medio de validación para la salida 1 es (ECM):", np.around(np.mean(ECM_fo),decimals=3),"+-",np.around(np.std(ECM_fo),decimals=3))
    print("El error medio absoluto de validación para la salida 1  es (MAE):", np.around(np.mean(MAE_fo),decimals=3),"+-",np.around(np.std(MAE_fo),decimals=3))
    print("El porcentaje de error medio absoluto de validación para la salida 1 es (MAPE):", np.around(np.mean(MAPE_fo),decimals=3),"%","+-",np.around(np.std(MAPE_fo),decimals=3),"%")
    print("Coeficiente de determinación para la salida 1 es (R2):", np.around(np.mean(R2_SCORE_fo),decimals=3),"%","+-",np.around(np.std(R2_SCORE_fo),decimals=3))

    print("El error cuadratrico medio de validación para la salida 2 es (ECM):", np.around(np.mean(ECM_so),decimals=3),"+-",np.around(np.std(ECM_so),decimals=3))
    print("El error medio absoluto de validación para la salida 2 es (MAE):", np.around(np.mean(MAE_so),decimals=3),"+-",np.around(np.std(MAE_so),decimals=3))
    print("El porcentaje de error medio absoluto de validación para la salida 2 es (MAPE):", np.around(np.mean(MAPE_so),decimals=3),"%","+-",np.around(np.std(MAPE_so),decimals=3),"%")
    print("Coeficiente de determinación para la salida 2 es (R2):", np.around(np.mean(R2_SCORE_so),decimals=3),"%","+-",np.around(np.std(R2_SCORE_so),decimals=3))