# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 10:16:30 2018

@author: win 10
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:26:12 2018

@author: win 10
"""

#This will perform the classification of heart disease by seperating the prediction values
#the result will be like 0 fpr absence amd 1for present
import pandas as pa
#from numpy import genfromtxt    #importing the data in general txt format
import numpy as np   #whenever v use np v r calling the library numpy
import matplotlib   #used for ploting
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC    #to import the opensource available toolbax for SVC
from sklearn.decomposition import PCA    #used for decomposition or feature extraction 
#import pylab as pl   #PCA-principal component analysis
from itertools import cycle
from sklearn import cross_validation #used to identify tat having disease or not
from sklearn.svm import SVC

#loading and printing the data
#dataset = genfromtxt('data_set.csv',dtype=float,delimiter=',')
dataset=pa.read_csv("C:\\dataset_final.csv")
#print dataset
X=dataset.iloc[:,0:13].values#feature set
Y=dataset.iloc[:,13].values#label set
#replacing 1-4 by 1 label
for index,item in enumerate(Y):
	if not(item == 0.0):
		Y[index]=1
print (Y)
target_names=['0','1']

#method to plot the graph for reduced dimentions
def plot_2D(data,target,target_names):
	colors=cycle('rgbcmykw')
	target_ids=range(len(target_names))
	plt.figure()
	for i,c, label in zip(target_ids,colors,target_names):
		plt.scatter(data[target==i,0],data[target==i,1],c=c,label=label)
		plt.legend()
		plt.savefld('Problem 2 graph')
#Classifying the data using Linear SVM and predicing the probability of disease belonging to a particular class

pca=PCA(n_components=2,whiten=True).fit(X)
X_new=pca.transform(X)
print(X_new)
#calling plot_20
plot_2D(X_new,Y,target_names)
#Applying cross validation on the training and set for validating our Linear SVM model
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.2,train_size=0.8,random_state=0)
modelSVM=LinearSVC()
modelSVM=modelSVM.fit(X_train,Y_train)
print("Linear SVC values with split")
print (modelSVM.score(X_test,Y_test))
modelSVMRaw=LinearSVC(C=0.1)
modelSVMRaw=modelSVMRaw.fit(X_new,Y)
cnt=0
for i in modelSVMRaw.predict(X_new):
	if i==Y[i]:
		cnt=cnt+1
print("linear SVC score without split")
print (float(cnt)/10) 
#using Stratified K Fold
skf=cross_validation.StratifiedKFold(Y,n_folds=2)
for train_index,test_index in skf:
	X_train3,X_test3=X[train_index],X[test_index]
	Y_train3,Y_test3=Y[train_index],Y[test_index]
print(X_train3)
print(Y_train3)
modelSVM3=SVC(C=0.1,kernel='rbf')
modelSVM3=modelSVM3.fit(X_train3,Y_train3)
print("Stratified K fold score")
print (modelSVM3.score(X_test3,Y_test3))
modelSVM3Raw=SVC(C=1.0,kernel='rbf')
modelSVM3Raw =modelSVM3Raw.fit(X_new,Y)
cnt2=0
for i in modelSVM3Raw.predict(X_new):
	if i==Y[i]:
		cnt2=cnt2+1
print("On PCA valued X_new")
print(cnt2/10)
#create a mesh to poltin 
X_min,X_max=X_new[:,0].min()-1,X_new[:,0].max()+1
Y_min,Y_max=X_new[:,1].min()-1,X_new[:,1].max()+1
xx,yy=np.meshgrid(np.arange(X_min,X_max,0.2),np.arange(Y_min,Y_max,0.2))
#title for the plots
titles='SVC (RBF kernel)-Plotting highest varied 2 PCA values'
#plot the decision boundries
plt.subplot(2,2,i+1)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
Z=modelSVM3.predict(np.c_[xx.ravel(),yy.ravel()])
#put the resultinto a color plot
Z=Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,alpha=0.8)
#Plot also the training points
plt.scatter(X_new[:,0],X_new[:,1],c=Y,cmap=plt.cm.Paired)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xticks(())
plt.yticks(())
plt.title(titles)
plt.show()


