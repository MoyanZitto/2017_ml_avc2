# The result of the code

cross validate 0:accuracy 0.775000

cross validate 1:accuracy 0.770000

cross validate 2:accuracy 0.825000

cross validate 3:accuracy 0.805000

cross validate 4:accuracy 0.835000

5-fold average accuracy: 0.802000


## The code of kNN

``` python
# Attention
# Not have much time to learn python
# just look through and copy other's code 
# will try to write the code on my own next time

import numpy as np

# read the data and type transformation
f=open('dataset.txt')
lines=f.readlines()

data=[]
label=[]


N=len(lines)
for i in range(N):
    a=lines[i].split()
    b=[float(item) for item in a[:3]]
    data.append(b)
    label.append(a[-1])

data=np.array(data)
label=np.array(label)


# choose the top nearest k  and count them
def knn(X1,y1,x2,k):
    n=len(X1)
    dist=[]
    for i in range(n):
        d=np.sqrt(np.sum((x2-X1[i])**2))
        dist.append(d)
    
    #得到距离最近的k个label
    k_index=np.argsort(dist)
    k_label=y1[k_index[:k]]
    
    #统计出现次数
    m=len(k_label)
    k_num=[]
    for j in range(m):
        k_num.append(list(k_label).count(k_label[j]))
    
    label=k_label[np.argmax(k_num)]
    
    return label
    
# 5-fold validation
k_fold=5

# split the data into five groups on average
data5=np.array_split(data,k_fold)
label5=np.array_split(label,k_fold)
accuracy=[]

# print data5
for i in range(k_fold):
    test_data=data5[i]
    test_label=label5[i]
	
    data4=[data5[j] for j in range(k_fold) if j!=i]
    label4=[label5[j] for j in range(k_fold) if j!=i]
	
    train_data=data4[0]
    train_label=label4[0]
	
    # print train_label.shape
    for m in range(1,4):
        train_data=np.vstack((train_data,data4[m]))
        train_label=np.hstack((train_label,label4[m]))
    # print train_data.shape
    # print train_label.shape
       
    y=[]
    m=len(test_data)

    for k in range(m):
        #使用的是4近邻
        out=knn(train_data,train_label,test_data[k],8)
        #print(y)
        #print(test_label[11])
        y.append(out)
		
    accuracy.append((np.sum(y==test_label))/m) 
	
    print('cross validate %d:accuracy %f'%(i,accuracy[i]))

print('5-fold average accuracy: %f'%np.mean(accuracy))  
```
