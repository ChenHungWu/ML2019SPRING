import csv 
import numpy as np
#from numpy.linalg import inv
#import random
import math
import sys
#import matplotlib.pyplot as plt
#from statistics import mean

degree = 1


#data_type=['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])
n_row = 0
text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")

for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))    
    n_row = n_row+1
text.close()

x = []
y = []
# 每 12 個月
for i in range(12):    
# 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                for m in range(1,degree+1):
                    if((data[t][480*i+j+s])<0):
                        x[471*i+j].append((data[t][480*i+j+s-1])**m)
                    else:
                        x[471*i+j].append((data[t][480*i+j+s])**m)
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

for i in range(len(x)):
    for j in range(0,len(x[i]),9*degree):
        for k in range(9*degree):
            if(x[i][j+k]<0):
                if i==0:
                    x[i][j+k] = np.average( [ x[i+1][j:j+degree*9:degree] ] )
                else:
                    x[i][j+k] = np.average( [ x[i-1][j:j+degree*9:degree] ] )



# add square term
# x = np.concatenate((x,x**2), axis=1)


#Normalization

mean = np.mean(x, axis = 0) 
std = np.std(x, axis = 0)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if not std[j] == 0 :
            x[i][j] = (x[i][j]- mean[j]) / std[j]

kill = []
for i in range(9):
    kill.append(18*i+0)
    kill.append(18*i+1)
    kill.append(18*i+2)
    kill.append(18*i+3)
    kill.append(18*i+4)
    kill.append(18*i+5)
    kill.append(18*i+6)
    kill.append(18*i+10)
    kill.append(18*i+11)
    kill.append(18*i+12)
    kill.append(18*i+13)
    kill.append(18*i+14)
    kill.append(18*i+15)
    kill.append(18*i+16)
    kill.append(18*i+17)

x = np.delete(x, kill, axis=1)

# add bias 在所有data前面加排1代表bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)



w = np.zeros(len(x[0]))



load_name = sys.argv[1]
#x = np.load(    'x.npy')
#y = np.load(    'y.npy')
'''
w = np.load(load_name+'_w.npy')
mt = np.load(load_name+'_mt.npy')
vt = np.load(load_name+'_vt.npy')
'''

print(x.shape)
print(y.shape)
print(w.shape)
w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(),x)),x.transpose()),y)
hypo = np.dot(x,w)
loss = hypo - y
cost = np.sum(loss**2) / len(x)
cost_a  = math.sqrt(cost)
print ('Cost: %.9f  ' % (cost_a))

# save model
#np.save(    'x.npy',x)
#np.save(    'y.npy',y)
'''
save_name = sys.argv[2]
np.save(save_name+'_w.npy',w)
np.save(save_name+'_mt.npy',mt)
np.save(save_name+'_vt.npy',vt)
'''

# read model
#w = np.load('model.npy')
#input test data
test_x = []
n_row = 0
text = open('data/test.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            for m in range(1,degree+1):
                test_x[n_row//18].append( (float(r[i])**m))
    else :
        for i in range(2,11):
            for m in range(1,degree+1):
                if r[i] !="NR":
                    test_x[n_row//18].append( (float(r[i])**m))
                else:
                    test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)
# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

for i in range(len(test_x)):
    for j in range(0,len(test_x[i]),degree*9):
        for k in range(degree*9):
            if(test_x[i][j+k]<0):
                if i == 0 :
                    test_x[i][j+k] = np.average( [ test_x[i+1][j:j+degree*9:degree] ] )
                else :
                    test_x[i][j+k] = np.average( [ test_x[i-1][j:j+degree*9:degree] ] )

for i in range(test_x.shape[0]):
    for j in range(test_x.shape[1]):
        if not std[j] == 0 :
            test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]

# add bias
test_x = np.delete(test_x, kill, axis=1)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = "result/degree"+str(degree)+"_"+str(cost_a)+".csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()


print('end')