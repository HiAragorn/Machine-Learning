# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:19:45 2018

@author: tresorom
"""


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the dataset 

dataset = pd.read_csv('Salary_Data.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

# splitting the dataset Train and Test==================================

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

# Fearure Scaling======================================================

'''from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

# For Simple Linear Regression, the model will take care of scaling 

#Fitting Simple Linear Regression to the Training Set=============

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

#===================================
# Predicting the test set results

y_pred = regressor.predict(x_test)#y_pred is the prediction and y_test is the real data 


#Visualizing the training and the results 
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color='blue' )
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

# Above, the blue line represents the prediction and the red the real salary

# Now we will test it

# Visualising the test set results

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')#this is the same than above, because it was already trained
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

























































