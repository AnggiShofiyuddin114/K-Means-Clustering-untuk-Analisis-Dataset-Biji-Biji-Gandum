import numpy as np
import csv
import math

with open('seeds_dataset.csv') as csvfile:
    csv1=csv.reader(csvfile)
    next(csv1,None)#skip header/judul
    dataset = list(csv1)
center=np.array([[14.64847222, 14.46041667,  0.87916667,  5.56377778,  3.27790278,
         2.64893333,  5.19231944],
       [18.72180328, 16.29737705,  0.88508689,  6.20893443,  3.72267213,
         3.60359016,  6.06609836],
       [11.96441558, 13.27480519,  0.8522    ,  5.22928571,  2.87292208,
         4.75974026,  5.08851948]])
x=[]
y=[]
for row in dataset:
    r=row[0].split(";")
    x.append(r[0:len(r)-1])
    y.append(int(r[len(r)-1])-1)
data=np.array(x,dtype='float32')
Y=np.array(y)
def random_test(x,y,test_size_class):
    pos,Class,random=[],[],True
    jumlah_class=[]
    for i in range(len(y)):
        if(y[i] not in Class):
            jumlah_class.append(0)
            pos.append(i)
            Class.append(y[i])
        elif((y[i] in Class) and y[i] != Class[-1]):
            random=False
            break
        jumlah_class[-1]+=1
    pos.append(i+1)
    if(not random):
        x_train=X[:round(len(x)*(1-test_size_class))]
        x_test=X[round(len(x)*(1-test_size_class)):]
        y_train=Y[:round(len(x)*(1-test_size_class))]
        y_test=Y[round(len(x)*(1-test_size_class)):]
    else:
        x_train,x_test,y_train,y_test=[[]],[[]],[[]],[[]]
        for i in range(len(Class)):
            rdm=pos[i]+round((pos[i+1]-pos[i])*(1-test_size_class))
            x_train+=list(x[pos[i]:rdm])
            x_test+=list(x[rdm:pos[i+1]])
            y_train+=list(y[pos[i]:rdm])
            y_test+=list(y[rdm:pos[i+1]])
        x_train.remove([])
        x_test.remove([])
        y_train.remove([])
        y_test.remove([])
    return np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)
def transpose(x):
    return np.transpose(x)
def KMean_clustering(data,centroid):
    kmean=[]
    dataAnggotaCluster=[ [] for _ in range(len(centroid))]
    for i in range(len(data)):
        ecludian=[]
        for j in range(len(centroid)):
            r=math.sqrt(sum((data[i]-centroid[j])**2))
            ecludian.append(r)
        kmean.append(ecludian.index(min(ecludian)))
        indek=ecludian.index(min(ecludian))
        dataAnggotaCluster[indek].append(data[i])
    list=[ [] for _ in range(len(centroid))]
    for i in range(len(centroid)):
        dataAnggotaCluster[i]=np.array(dataAnggotaCluster[i])
        total=np.array([ 0 for _ in range(len(data[0]))],dtype='float32')
        for j in range(len(dataAnggotaCluster[i])):
            total+=dataAnggotaCluster[i][j]
        list[i]=np.array(total/len(dataAnggotaCluster[i]))
    return kmean,np.array(list),dataAnggotaCluster
def KMean(data,center):
    kmean_clustering=[]
    while True:
        kmean,center,dataAnggotaCluster=KMean_clustering(data,center)
        if(kmean_clustering==kmean): break
        kmean_clustering=kmean
    return center,kmean,dataAnggotaCluster
def jarakRata2ObjekCluster(objek,objeks):
    jarak=transpose(sum(transpose((objek-objeks)**2))**0.5)
    return np.mean(jarak)
def Testing(data,centroid):
    kmean=[]
    dataAnggotaCluster=[ [] for _ in range(len(centroid))]
    for i in range(len(data)):
        ecludian=[]
        for j in range(len(centroid)):
            r=math.sqrt(sum((data[i]-centroid[j])**2))
            ecludian.append(r)
        kmean.append(ecludian.index(min(ecludian)))
        indek=ecludian.index(min(ecludian))
        dataAnggotaCluster[indek].append(data[i])
    return kmean,dataAnggotaCluster
def Silhoutte(anggota_cluster):
    Silhoutte=[]
    for i in range(len(anggota_cluster)):
        for j in range(len(anggota_cluster[i])):
            anggotaCluster_a=np.array(list(anggota_cluster[i][:j])+list(anggota_cluster[i][j+1:]))
            jarak_a=jarakRata2ObjekCluster(anggota_cluster[i][j],anggotaCluster_a)
            jarak_b=[]
            for k in range(len(anggota_cluster)):
                if(k!=i):
                    jarak=jarakRata2ObjekCluster(anggota_cluster[i][j],anggota_cluster[k])
                    jarak_b.append(jarak)
            if(jarak_a<min(jarak_b)):
                silhoutte=1-jarak_a/min(jarak_b)
            else:
                silhoutte=min(jarak_b)/jarak_a-1
            Silhoutte.append(silhoutte)
    return np.mean(Silhoutte)
x_train,x_test,y_train,y_test=random_test(data,Y,0.4)
new_center,kmeanTrain,dataAnggotaClusterTrain=KMean(x_train,center)
kmeanTest,dataAnggotaClusterTest=Testing(x_test,new_center)
silhouette=Silhoutte(dataAnggotaClusterTest)
print('cluster center =')
print(center)
print('hasil testing =')
print(kmeanTest)
print('silhouette coefficient = ',silhouette)

##from sklearn import metrics
##silhouette=metrics.silhouette_score(x_test, kmeanTest,metric='euclidean')
##print(silhouette)
