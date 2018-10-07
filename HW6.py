# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 15:01:10 2018

@author: hongy
"""

from sklearn import datasets 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score 

print("My name is Hongyu Zhang")
print("My NetID is: hongyuz2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

iris = datasets.load_iris() 
X = iris.data[:, [2, 3]] 
y = iris.target 
 

#decision tree
tree = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)

rmax=11
r_range=range (1,rmax)
accuracy_train=[]
accuracy_test=[]
cv_train=[]
cv_test=[]
rs=[]
for r in r_range: 
    rs.append(r)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=r)
    #standard 
    sc = StandardScaler() 
    sc.fit(X_train) 
    X_train_std = sc.transform(X_train) 
    X_test_std = sc.transform(X_test)

    tree.fit(X_train_std,y_train)
    y_train_pred=tree.predict(X_train_std) 
    y_test_pred = tree.predict(X_test_std) 
    
    #accuracy
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))  
    

#print the accuracy in different random state
print('\n'*3)
print("in-sample accuracy")
print(accuracy_train)
plt.plot(rs,accuracy_train,'bo')
plt.show()
print("Mean of in-sample accuracy:{}".format(np.mean(accuracy_train)))
print("standard deviation of in-sample accuracy:{}".format(np.std(accuracy_train)))

print('\n'*3)
print("out-of-sample accuracy")
print(accuracy_test)
plt.plot(rs,accuracy_test,'bo')
plt.show()  
print("Mean of out-of-sample accuracy:{}".format(np.mean(accuracy_test)))
print("standard deviation of out-of-sample accuracy:{}".format(np.std(accuracy_train)))


#cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=4)

tree.fit(X_train,y_train)
y_train_pred=tree.predict(X_train) 
y_test_pred = tree.predict(X_test)  
cv_train=cross_val_score(tree, X_train, y_train_pred,cv=10) 
cv_test=cross_val_score(tree, X_test, y_test_pred,cv=10)
x_axis=range(1,11)
print('\n'*3)
print('in-sample cross_val_score')
print(cv_train)
plt.plot(x_axis,cv_train,'bo')
plt.show()
print("Mean of in-sample cross_val_score:{}".format(np.mean(cv_train)))
print("standard deviation of in-sample cross_val_score:{}".format(np.std(cv_train)))

print('\n'*3)
print('out-of-sample cross_val_score')
print(cv_test)
plt.plot(x_axis,cv_test,'bo')
plt.show()  
print("Mean of out-of-sample cross_val_score:{}".format(np.mean(cv_test)))
print("standard deviation of out-of-sample cross_val_score:{}".format(np.std(cv_test)))
