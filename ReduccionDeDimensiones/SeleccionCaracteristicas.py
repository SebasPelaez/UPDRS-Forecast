import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def SFS(X,Y):
    N,M = np.shape(X)
    indices = list(np.arange(M))
    high_score = 0
    index_max = []
    flag = True
    while flag:
      score = []
      for i in range(0,len(indices)):
        aux_X = index_max + [indices[i]]
        X_train, X_test, y_train, y_test = train_test_split(X[:,aux_X], Y, test_size = 0.2, random_state = 0)
        regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
        regressor.fit(X_train, y_train)
        score.append(regressor.score(X_test,y_test))
    
      b = np.argmax(score)
      if (score[b] > high_score):
        high_score = score[b]
        index_max.append(indices[b])
        indices.pop(b)
      else:
        flag = False
      
    return(index_max)
        
        

