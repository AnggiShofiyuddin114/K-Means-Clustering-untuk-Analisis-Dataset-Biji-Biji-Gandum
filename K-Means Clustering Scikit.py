from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import csv
from sklearn.model_selection import train_test_split

with open('seeds_dataset.csv') as csvfile:
    csv1=csv.reader(csvfile)
    next(csv1,None)#skip header/judul
    dataset = list(csv1)
x=[]
y=[]
for row in dataset:
    r=row[0].split(";")
    x.append(r[0:len(r)-1])
    y.append(int(r[len(r)-1])-1)
X=np.array(x)
Y=np.array(y)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4, random_state=20)
kmeans=KMeans(n_clusters=3)
kmeans.fit(x_train)
y_kmeans=kmeans.predict(x_test)
silhouette=metrics.silhouette_score(x_test, y_kmeans,metric='euclidean')
print('cluster center =')
print(kmeans.cluster_centers_)
print('hasil testing =')
print(y_kmeans)
print('silhouette coefficient = ',silhouette)
