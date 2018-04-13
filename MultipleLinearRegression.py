import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def Regression(X_train,X_test,Y_train,Y_test):
    y_train_fo = Y_train[:,0]
    y_train_so = Y_train[:,1]
    y_test_fo = Y_test[:,0]
    y_test_so = Y_test[:,1]
    
    #Normalización de los datos  
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    fo = LinearRegression()
    so = LinearRegression()
      
    fo.fit(X_train, y_train_fo)
    so.fit(X_train, y_train_so)
      
    # Predecir los resultados de prueba
    y_pred_fo = fo.predict(X_test)
    y_pred_so = so.predict(X_test)
    
    y_test = np.concatenate((y_test_fo,y_test_so),axis=1)
    y_pred = np.concatenate((y_pred_fo,y_pred_so),axis=1)
    
    return y_test,y_pred