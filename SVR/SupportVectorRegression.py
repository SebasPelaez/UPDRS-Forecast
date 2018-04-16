import numpy as np

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor

def SupportVectorRegression(X,Y,groups,kernel_value):
    lpgo = GroupKFold(n_splits=14)
    MAE = []
    ECM = []
    N = np.size(Y[0])
    for train_index, test_index in lpgo.split(X, Y, groups):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = Y[train_index], Y[test_index]
      
      #Normalización de los datos      
      sc_X = StandardScaler()
      X_train = sc_X.fit_transform(X_train)
      X_test = sc_X.transform(X_test)
      
      # Ajustar el modelo
      svr_lineal = SVR(kernel = kernel_value)
      multiple_output_regressor = MultiOutputRegressor(svr_lineal)
      
      multiple_output_regressor.fit(X_train, y_train)
      
      y_pred = multiple_output_regressor.predict(X_test)
      
      ECM.append(mean_squared_error(y_test,y_pred,multioutput='raw_values'))
      MAE.append(mean_absolute_error(y_test,y_pred,multioutput='raw_values'))
      
    ECM_matrix = np.asmatrix(ECM)
    MAE_matrix = np.asmatrix(MAE)
    for i in range(0,N):
        print("El error cuadratrico medio de validación para la salida", (i+1),"es (ECM):", np.mean(ECM_matrix[:,i]))
        print("El error medio absoluto de validación para la salida", (i+1),"es (MAE):", np.mean(MAE_matrix[:,i]))