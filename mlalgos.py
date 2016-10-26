from sklearn import linear_model
import numpy as np
import pandas
import math
import matplotlib.pyplot as plt


#importing training and test data.
data = pandas.read_csv('train(1).csv')
test=pandas.read_csv('test.csv')

new_test=[]
new_age=[]
data=np.array(data)
test=np.array(test)

#Conversion into float.
test=test[0::,4].astype(np.float)
titanic_age=data[0::,5].astype(np.float)

#Filling missing data with 0s.
for e in range(0,len(titanic_age)):
    if math.isnan(titanic_age[e]):
        new_age=np.append(new_age,0)
    else:
        new_age=np.append(new_age,titanic_age[e])

for i in range(0,len(test)):
    if math.isnan(test[i]):
        new_test=np.append(new_test,0)
    else:
        new_test=np.append(new_test,test[i])


print(new_test)
#Reshape into similar dimensions.
x_test=new_test.reshape(-1,1)
x_train=new_age.reshape(-1,1)
y_train=data[0::,1].reshape(-1,1)
print(y_train)
plt.plot(x_train,'ro',y_train,'y-')
#plt.show()
#Linear Regression
linear=linear_model.LinearRegression()
linear.fit(x_train,y_train)
linear.score(x_train,y_train)
predicted=linear.predict(x_test)


#print(predicted)
