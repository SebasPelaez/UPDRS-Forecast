import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

def RandomForest(X,Y,groups,n_trees):
    lpgo = GroupKFold(n_splits=14)
    MAE = []
    ECM = []
    MAPE = []
    R2_SCORE = []
    relevant_features = []
    N = np.size(Y[0])
    for train_index, test_index in lpgo.split(X, Y, groups):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = Y[train_index], Y[test_index]
    
      #Normalización de los datos
      sc_X = StandardScaler()
      X_train = sc_X.fit_transform(X_train)
      X_test = sc_X.transform(X_test)
    
      # Ajustar el modelo a la regresión simple
      regressor = RandomForestRegressor(n_estimators = n_trees, random_state = 0)
      regressor.fit(X_train, y_train)
      
      relevant_features.append(regressor.feature_importances_)
    
      # Predecir los resultados de prueba
      y_pred = regressor.predict(X_test)
    
      ECM.append(mean_squared_error(y_test,y_pred,multioutput='raw_values'))
      MAE.append(mean_absolute_error(y_test,y_pred,multioutput='raw_values'))
      R2_SCORE.append(r2_score(y_test, y_pred, multioutput='raw_values'))
      m = []
      m.append(np.mean(np.abs((y_test[:,0] - y_pred[:,0]) / y_test[:,0])) * 100)
      m.append(np.mean(np.abs((y_test[:,1] - y_pred[:,1]) / y_test[:,1])) * 100)
      MAPE.append(m)
    
    ECM_matrix = np.asmatrix(ECM)
    MAE_matrix = np.asmatrix(MAE)
    MAPE_matrix = np.asmatrix(MAPE)
    R2_matrix = np.asmatrix(R2_SCORE)
       
    for i in range(0,N):
        print("El error cuadratrico medio de validación para la salida", i,"es (ECM):", np.mean(ECM_matrix[:,i]))
        print("El error medio absoluto de validación para la salida", i,"es (MAE):", np.mean(MAE_matrix[:,i]))
        print("El porcentaje de error medio absoluto de validación para la salida", (i+1),"es (MAPE):", np.mean(MAPE_matrix[:,i]),"%")
        print("Coeficiente de determinación para la salida", (i+1),"es (R2):", np.around(np.mean(R2_matrix[:,i])),"%","+-",np.around(np.std(R2_matrix[:,i]),decimals=5))
    
    relevant_features = np.asmatrix(relevant_features)
    print (np.mean(relevant_features,axis = 0))