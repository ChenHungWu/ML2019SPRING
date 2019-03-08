import csv 
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

degree = 2


#data_type=['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
#data input test.csv result.csv train.csv
test_name = sys.argv[1]
result_name = sys.argv[2]
x = []

if(len(sys.argv)>3):
    train_name = sys.argv[3]
    data = []
    # 每一個維度儲存一種污染物的資訊
    for i in range(18):
        data.append([])
    n_row = 0
    text = open(train_name, 'r', encoding='big5') 
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
    #把小於0的data換成平均
    for i in range(len(x)):
        for j in range(0,len(x[i]),9*degree):
            for k in range(9*degree):
                if(x[i][j+k]<0):
                    if i==0:
                        x[i][j+k] = np.average( [ x[i+1][j:j+degree*9:degree] ] )
                    else:
                        x[i][j+k] = np.average( [ x[i-1][j:j+degree*9:degree] ] )


    #Normalization 可是效果不好
    '''
    mean = np.mean(x, axis = 0) 
    std = np.std(x, axis = 0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not std[j] == 0 :
                x[i][j] = (x[i][j]- mean[j]) / std[j]
    '''
    '''
    reverse_column = []
    for i in range(18):
    	reverse_column.append(i*9+8)
    	reverse_column.append(i*9+7)
    	reverse_column.append(i*9+6)
    	reverse_column.append(i*9+5)
    	reverse_column.append(i*9+4)
    reverse_column = range(81,90)
    x = x[:,reverse_column]
    '''
    # add bias 在所有data前面加排1代表bias

    
    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
    w = np.zeros(len(x[0]))
    mt = np.zeros(len(x[0]))
    vt = np.zeros(len(x[0]))
   
    
    #Adam 40萬 0.0001
    l_rate = 0.0001 
    repeat = 400
    b1 = 0.9
    b2 = 0.999
    e = 1e-8
    x_t = x.transpose()
    #lamda = [5,3,1,0.1,0.01,0.001,0.0001]
    #color = ['red','green','blue','gray','peru','g','gold']
    
    
    for i in range(1,repeat):
        #calculate gradient
        diff = np.dot(x,w)
        loss = diff - y
        #regularization
        #loss += lamda[t] * np.sum((np.concatenate((np.array([0]),w[1:]),axis=0) **2 ))
        gra = 2.0 * np.dot(x_t,loss) #+ 2*lamda[t]*(np.concatenate((np.array([0]),w[1:]),axis=0))
        #update weight
        mt = b1*mt + (1-b1)*gra
        vt = b2*vt + (1-b2)*(gra*gra)
        m_cap = mt / (1-(b1**i))
        v_cap = vt / (1-(b2**i))
        w = w - (l_rate * m_cap) / (np.sqrt(v_cap)+e) 
        #calculate cost
        cost = np.sum(loss**2) / len(x)
        cost_a  = math.sqrt(cost)
        print ('iteration: %d | Cost: %.9f  ' % ( i,cost_a))
   # np.save('my_w.npy',w)
   # np.save('my_mt.npy',mt)
   # np.save('my_vt.npy',vt)



if(len(sys.argv)==3):
    w = np.load('my_w.npy')
    mt = np.load('my_mt.npy')
    vt = np.load('my_vt.npy')
# read model
#w = np.load('model.npy')
#input test data
test_x = []
n_row = 0
text = open(test_name ,"r")
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

#把小於0的data換成平均
for i in range(len(test_x)):
    for j in range(0,len(test_x[i]),degree*9):
        for k in range(degree*9):
            if(test_x[i][j+k]<0):
                if i == 0 :
                    test_x[i][j+k] = np.average( [ test_x[i+1][j:j+degree*9:degree] ] )
                else :
                    test_x[i][j+k] = np.average( [ test_x[i-1][j:j+degree*9:degree] ] )

#Normalization 可是效果不好
'''
for i in range(test_x.shape[0]):
    for j in range(test_x.shape[1]):
        if not std[j] == 0 :
            test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]
'''
#test_x = test_x[:,reverse_column]
# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = result_name
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()


print('end')