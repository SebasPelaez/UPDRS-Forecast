from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

def SupportVectorRegression(X,Y,groups,kernel_value):
    lpgo = GroupKFold(n_splits=14)
    MAE = []
    ECM = []
    for train_index, test_index in lpgo.split(X, Y, groups):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = Y[train_index], Y[test_index]
      
      #Normalización de los datos      
      sc_X = StandardScaler()
      X_train = sc_X.fit_transform(X_train)
      X_test = sc_X.transform(X_test)
      
      # Ajustar el modelo
      svr_lineal = SVR(kernel = kernel_value)
      
      svr_lineal.fit(X_train, y_train)
      
      y_pred = svr_lineal.predict(X_test)
      
      ECM.append(mean_squared_error(y_test,y_pred))
      MAE.append(mean_absolute_error(y_test,y_pred))
      
    print("El error cuadratrico medio de validación es (ECM):", np.mean(ECM))
    print("El porcentaje de error medio absoluto de validación es (MAE):", np.mean(MAE))