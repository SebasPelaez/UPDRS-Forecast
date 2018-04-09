# Importing las bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Organizar las variables de entrada y las variables a predecir
full_data = pd.read_csv('/home/sebas/Escritorio/Proyecto Simulación/Data.csv')
x1 = full_data.iloc[:, 0:4].values
x2 = full_data.iloc[:, 6:22].values
X = np.concatenate((x1,x2),axis=1)
y = full_data.iloc[:, 4:6].values

#Normalización de los datos
sc_X = StandardScaler() #Creo el objeto
#X_train = sc_X.fit_transform(X_train) #Normalizo los datos de entrenamiento
#X_test = sc_X.transform(X_test) #Normalizo los datos de prueba con base en los de entrenamiento

for fold in range(1,10):
    #Se realiza el proceso de validación cruzada
    #Método Leave-One-Speaker-Out
    print("Validación Cruzada")
    #Se entrena el modelo
    #Usar cada uno de los modelos que se van a entrenar
    print("Entrenamiento del modelo")
    #Se prueba el modelo
    #Se valida cada uno de los modelos que se entrenaron
    print("Validación del modelo")
