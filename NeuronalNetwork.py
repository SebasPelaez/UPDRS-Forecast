from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

def NeuronalNetwork(X,Y,groups):    
    lpgo = GroupKFold(n_splits=5)
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
      neuronal_network = MLPRegressor(hidden_layer_sizes=(20, ),random_state=0,max_iter=90)
      
      y_pred = neuronal_network.fit(X_train, y_train).predict(X_test)    
      
     
      ECM.append(mean_squared_error(y_test,y_pred))
      MAE.append(mean_absolute_error(y_test,y_pred))
      
    print("El error cuadratrico medio de validación es (ECM):", np.mean(ECM))
    print("El porcentaje de error medio absoluto de validación es (MAE):", np.mean(MAE))