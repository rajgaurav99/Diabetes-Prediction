import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


X=pd.read_csv('d2.csv')
X=pd.DataFrame(X)

#kmeans = KMeans(n_clusters=2, random_state=0, init=np.array([[0,0],[189,846]]), n_init=1).fit(X)
kmeans = KMeans(n_clusters=2,init=np.array([[0,0],[189,846]]),n_init=1) 
y_kmeans = kmeans.fit_predict(X)
centres=kmeans.cluster_centers_
centrex=[]
centrey=[]
for i in range(len(centres)):
    centrex.append(centres[i][0])
    centrey.append(centres[i][1])
color=['red','green','blue','purple','black']
print('Clusters formed: ')
print(y_kmeans)
for i in range(len(X)):
    plt.scatter(X.iloc[i,0],X.iloc[i,1],color=color[y_kmeans[i]])
plt.scatter(centrex,centrey,color='yellow',s=300,label='Cluster Centroid')
plt.xlabel('Glucose')
plt.ylabel('Insulin')   
plt.legend()
plt.title(' Clusters visualized from different colours') 
plt.show()
