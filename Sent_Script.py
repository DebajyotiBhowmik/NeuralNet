#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 22:16:23 2017

@author: debajyoti
"""

import numpy as np
import sys
from sklearn.utils import shuffle

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_data():
    mu1=np.array([10,10])
    sigma1=np.array([[6,1],[1,6]])
    
    mu2=np.array([20,20])
    sigma2=np.array([[5,2],[2,5]])
    
    sample_class1=np.random.multivariate_normal(mu1,sigma1,50)
    sample_class2=np.random.multivariate_normal(mu2,sigma2,50)
    
    result=np.concatenate((sample_class1,sample_class2),axis=0)
    
    #plt.legend(loc='upper left');
    
    return result
def calculate_weight(X_train,max_c1,max_c2):
    class_1=X_train[0:max_c1,:]
    class_2=X_train[max_c1:max_c2,:]
    ############## check which one is class one ##########
    class_1_mean=np.reshape(np.mean(class_1,axis=0),newshape=[-1,1])
    class_2_mean=np.reshape(np.mean(class_2,axis=0),newshape=[-1,1])
    
    class_1_mean_norm=np.dot(class_1_mean.T,class_1_mean)
    class_2_mean_norm=np.dot(class_2_mean.T,class_2_mean)
    print(class_1_mean_norm,class_2_mean_norm,sep='  ')    
    if class_1_mean_norm > class_2_mean_norm:
        temp=class_1
        class_1=class_2
        class_2=temp
        print('changed')
    ########################add -1 (x0)###############################
    extra_vector_with_neg_one=np.ndarray([50,1])
    extra_vector_with_neg_one.fill(-1)
    class_1=np.array(class_1)
    class_2=np.array(class_2)
    
    class_1=np.concatenate((class_1,extra_vector_with_neg_one),axis=1)
    class_2=np.concatenate((class_2,extra_vector_with_neg_one),axis=1)
    class_1=shuffle(class_1)
    class_2=shuffle(class_2)
    
    lenght_feature=X_train.shape[1]
    weight_vector=np.ndarray([3,1])
    weight_vector.fill(0.05)
    list_weight_vector=[]
    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #X_train=shuffle(X_train)
    update=1
    iteration=1
    eta=1
    list_weight_vector.append(weight_vector)
    while update==1 and iteration<=10000:        
        update=0
        iteration=iteration+1
        for i in range(class_1.shape[0]):
            point_vector=np.reshape(class_1[i], newshape=[1,-1]).T
            if np.dot(weight_vector.T,point_vector)>=0:
                weight_vector=weight_vector-eta*point_vector
                #print('misclassified')
                update=1  
    
        for i in range(class_2.shape[0]):
            point_vector=np.reshape(class_2[i], newshape=[1,-1]).T
            if np.dot(weight_vector.T,point_vector)<=0:
                weight_vector=weight_vector+eta*point_vector
                #print('misclassified 2')
                update=1
        list_weight_vector.append(weight_vector)  
        
    return weight_vector,iteration,list_weight_vector
def update(i):
    #####result is a sect of points (training)###########
    if i==iteration:
        exit
    label = 'timestep {0}'.format(i)
    weight_iter=list_weight_vector[i]
    
    ax1.set_xlabel(label) 
    low_point=(weight_iter[2][0]-weight_iter[0][0]*0)/weight_iter[1][0]
    high_point=(weight_iter[2][0]-weight_iter[0][0]*max(result[:,0]))/weight_iter[1][0]
    line.set_ydata([low_point,high_point])
    return line, ax1
    

result=generate_data()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(result[0:50,0],result[0:50,1],c='b',marker="o", label='first')# c='b', marker="s", label='first'
ax1.scatter(result[50:100,0],result[50:100,1],c='r',marker="o", label='second')

weight,iteration,list_weight_vector=calculate_weight(result,50,100)
print("iteration= ",iteration)

weight0=list_weight_vector[0]
lowest_point=[0,(weight0[2][0]-weight0[0][0]*0)/weight0[1][0]]
high_point=[max(result[:,0]),(weight0[2][0]-weight0[0][0]*max(result[:,0]))/weight0[1][0]]
line,=ax1.plot([lowest_point[0],high_point[0]],[lowest_point[1],high_point[1]])
#plt.show()

anim = FuncAnimation(fig, update, frames=np.arange(0, iteration), interval=100,repeat=False)
if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('line.gif', dpi=80, writer='imagemagick')
else:
        # plt.show() will just loop the animation forever.
        plt.show()

