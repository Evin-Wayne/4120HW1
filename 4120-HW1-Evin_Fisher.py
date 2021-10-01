#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#4120 HW1 Evin Fisher


# In[5]:


#Imports
import pandas as pd
import random
import matplotlib.pyplot as mplot
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
import statistics
#load iris data
iris_data = load_iris()


# In[6]:


#random pick
k1 = random.randint(1, 20)
k2 = random.randint(1, 20)
k3 = random.randint(1, 20)
k4 = random.randint(1, 20)
k5 = random.randint(1, 20)
k6 = random.randint(1, 20)
k7 = random.randint(1, 20)
print(k1)
print(k2)
print(k3)
print(k4)
print(k5)
print(k6)
print(k7)
print()
#find accuracy of K values
X= iris_data.data
Y= iris_data.target

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=0)

accSomeK = []

#k1
rK1 = KNeighborsClassifier(n_neighbors=k1)
rK1.fit(xTrain, yTrain)
yPrediction = rK1.predict(xTest)
acc = accuracy_score(yTest, yPrediction)
accSomeK.append(accuracy_score(yTest, yPrediction))
print(k1)
print("Accuracy:", acc)

#k2
rK2 = KNeighborsClassifier(n_neighbors=k2)
rK2.fit(xTrain, yTrain)
yPrediction = rK2.predict(xTest)
acc = accuracy_score(yTest, yPrediction)
accSomeK.append(accuracy_score(yTest, yPrediction))
print(k2)
print("Accuracy:", acc)

#k3
rK3 = KNeighborsClassifier(n_neighbors=k3)
rK3.fit(xTrain, yTrain)
yPrediction = rK3.predict(xTest)
acc = accuracy_score(yTest, yPrediction)
accSomeK.append(accuracy_score(yTest, yPrediction))
print(k3)
print("Accuracy:", acc)

#k4
rK4 = KNeighborsClassifier(n_neighbors=k4)
rK4.fit(xTrain, yTrain)
yPrediction = rK4.predict(xTest)
acc = accuracy_score(yTest, yPrediction)
accSomeK.append(accuracy_score(yTest, yPrediction))
print(k4)
print("Accuracy:", acc)

#k5
rK5 = KNeighborsClassifier(n_neighbors=k5)
rK5.fit(xTrain, yTrain)
yPrediction = rK5.predict(xTest)
acc = accuracy_score(yTest, yPrediction)
accSomeK.append(accuracy_score(yTest, yPrediction))
print(k5)
print("Accuracy:", acc)

#k6
rK6 = KNeighborsClassifier(n_neighbors=k6)
rK6.fit(xTrain, yTrain)
yPrediction = rK6.predict(xTest)
acc = accuracy_score(yTest, yPrediction)
accSomeK.append(accuracy_score(yTest, yPrediction))
print(k6)
print("Accuracy:", acc)

#k7
rK7 = KNeighborsClassifier(n_neighbors=k7)
rK7.fit(xTrain, yTrain)
yPrediction = rK7.predict(xTest)
acc = accuracy_score(yTest, yPrediction)
accSomeK.append(accuracy_score(yTest, yPrediction))
print(k7)
print("Accuracy:", acc)

#average accuracy
avg = statistics.mean(accSomeK)
print("The average accuracy is:", avg*100, "%.")


# In[7]:


#accuracy of all K's 1-20
X= iris_data.data
Y= iris_data.target

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=0)

#KNN 1-20 with average accuracy
allKRange = range(1,21)
accHere = {}
accAllK = []
for k in allKRange:
    allK = KNeighborsClassifier(n_neighbors=k)
    allK.fit(xTrain, yTrain)
    yPrediction = allK.predict(xTest)
    accHere[k] = accuracy_score(yTest, yPrediction)
    accAllK.append(accuracy_score(yTest, yPrediction))
avg = statistics.mean(accAllK)
print("The average accuracy is: ", avg*100, "%.")


# In[8]:


#line chart

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('default')
import numpy as np

fig = plt.figure()
ax = plt.axes()

X = np.array(range(1,21))
Y = np.array(accAllK)

plt.plot(X, Y, color='red')
plt.xlabel('Values of K')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy in Relation to Values of K for KNN')
plt.show()

#model accuracy
avg_value = statistics.mean(Y)
print("The average accuracy is: ", avg_value*100, "%.")


# In[ ]:




