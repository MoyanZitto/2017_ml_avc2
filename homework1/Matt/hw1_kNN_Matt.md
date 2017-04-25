# The result of the code

The accuracy of the dataset0 is 77.0%
The accuracy of the dataset1 is 79.0%
The accuracy of the dataset2 is 84.0%
The accuracy of the dataset3 is 80.5%
The accuracy of the dataset4 is 85.0%
The accuracy of the whole dataset is 81.1%

**结果的正确率不是特别高，可能是因为三个特征之间量级不一样，可以考虑特征归一化，再利用kNND分类**

## The code of kNN

``` python
# mywork
import numpy as np

# 读取数据
fn = 'dataset.txt'
fr = open(fn)

lines = fr.readlines()
datalen = len(lines)

data = []
label = []
for i in range(datalen):
	# 去掉头和尾的空格并分割字符
	line = lines[i].strip()	
	item = line.split('\t')
	# data表示特征x
	b = [float(temp) for temp in item[:3]]
	data.append((b))
	# label表示特征y
	label.append(item[-1])
	
fr.close()

# 转换成array
data = np.array(data)
label = np.array(label)

# 给定训练数据和测试数据，根据kNN进行分类
def kNN(traindata,trainlabel,testdata,k):
	
	datalen = len(traindata) 
	dist = []
	for i in range(datalen):
		temp = np.sqrt(np.sum((testdata-traindata[i])**2))  
		dist.append(temp)
	
	k_index = np.argsort(dist)
	k_label = trainlabel[k_index[:k]]
	k_label_num = []
	for i in range(k):
		k_label_num.append(list(k_label).count(k_label[i]))
	
	return k_label[np.argmax(k_label_num)]

k_fold = 5
# 将数据集分成五份
data_k_fold = np.array_split(data,k_fold)
label_k_fold = np.array_split(label,k_fold)


testsetlen = len(label_k_fold[0]) 
# 将每一类的数据作为测试集，将其他数据作为训练集，得到结果

accuracy = []

for i in range(k_fold):	

	# 去除其他类的list，list生成器
	traindata = [data_k_fold[j] for j in range(k_fold) if j!= i]
	trainlabel = [label_k_fold[j] for j in range(k_fold) if j!= i]

	# 合并成一个list 	
	traindata = np.vstack(traindata)
	trainlabel = np.hstack(trainlabel)
	
	correct = 0;
	for j in range(testsetlen):
		testdata = data_k_fold[i][j]
		output = kNN(traindata,trainlabel,testdata,15)
		if output == label_k_fold[i][j] :
			correct = correct + 1
	
	accuracy.append(correct * 100 / testsetlen)
	
	print("The accuracy of the dataset%s is %s%%" %(i,accuracy[i]))
	

print("The accuracy of the whole dataset is %s%%" %np.mean(accuracy))
```
