# -*- coding: utf-8 -*-  
import numpy as np  

filename = 'dataset.txt' 
data= []
label = []
with open(filename, 'r') as file_to_read:
  while True:
    lines = file_to_read.readline() # 整行读取数据
    if not lines:
      break
      pass
    a=lines.split()
    b=[float(i) for i in a[:3]]
    data.append(b)
    label.append(a[-1]) # 添加新读取的数据,这里有点不明白，分了只剩字符串？
    pass
   
  data = np.array(data) # 将数据从list类型转换为array类型。
  label = np.array(label)


#提取飞行里程
miles= range(0, 1000)
b=-1
while(b<0): 
  a=0
  b=b+1
  while(a<1000):
   miles[a]=data[a][b]
   a=a+1 

#提取玩游戏时间
time= range(0, 1000)
b=0
while(b<1): 
  a=0
  b=b+1
  while(a<1000):
   time[a]=data[a][b]
   a=a+1 

#提取吃冰糕数量
ice= range(0, 1000)
b=1
while(b<2): 
  a=0
  b=b+1
  while(a<1000):
   ice[a]=data[a][b]
   a=a+1 
 
#（0,1）归一化数据
a=0
min=np.min(miles)#里程归一化
max=np.max(miles)
while (a<1000):
 miles[a] = (miles[a] - min) / (max - min)
 a=a+1

a=0
min=np.min(time)#时间归一化
max=np.max(time)
while (a<1000):
 time[a] = (time[a] - min) / (max - min)
 a=a+1 
  
a=0
min=np.min(ice)#食量归一化
max=np.max(ice)
while (a<1000):
 ice[a] = (ice[a] - min) / (max - min)
 a=a+1  
  
#验证knn准确率，300训练集 未交叉验证
a=0
c=0
distance= range(0, 300)
while(a<300):
 b=0
 while(b<300): 
   distance[b]=(miles[a]-miles[b])*(miles[a]-miles[b])+ (time[a]-time[b])*(time[a]-time[b])+(ice[a]-ice[b])*(ice[a]-ice[b])#欧氏距离未开方
   b=b+1
   if a==b:
    distance[b]=1

 k_index=np.argsort(distance)
 k_label=label[k_index[:10]]
 a=a+1

 m=len(k_label)
 k_num=[]
 for j in range(m):
   k_num.append(list(k_label).count(k_label[j]))
   label_guji=k_label[np.argmax(k_num)]

 if label_guji==k_label[0]:
   c=c+1
accuracy=c/300.0
print accuracy

#验证knn准确率,700验证集 未交叉验证 
a=300
c=0
distance= range(0, 700)
while(a<1000):
 b=300
 while(b<1000): 
   distance[b-300]=(miles[a]-miles[b])*(miles[a]-miles[b])+ (time[a]-time[b])*(time[a]-time[b])+(ice[a]-ice[b])*(ice[a]-ice[b])#欧氏距离未开方
   b=b+1
   if a==b:
    distance[b-300]=1
 k_index=np.argsort(distance)
 k_label=label[300+k_index[:10]]
 a=a+1

 m=len(k_label)
 k_num=[]
 for j in range(m):
   k_num.append(list(k_label).count(k_label[j]))
   label_guji=k_label[np.argmax(k_num)]

 if label_guji==k_label[0]:
   c=c+1
accuracy=c/700.0
print accuracy

  
