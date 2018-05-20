import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def CoeficientePearson(X,Y):
    coeficientesCorrelacion_fo = np.corrcoef(X,Y[:,0],rowvar=False)
    coeficientesCorrelacion_so = np.corrcoef(X,Y[:,1],rowvar=False)
    
    plt.subplots(figsize=(15,10))
    ax1 = sns.heatmap(coeficientesCorrelacion_fo,cmap=plt.cm.jet,annot=True)
    
    plt.subplots(figsize=(15,10))
    ax2 = sns.heatmap(coeficientesCorrelacion_so,cmap=plt.cm.jet,annot=True)