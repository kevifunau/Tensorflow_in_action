#coding:utf-8

import tensorflow as tf 
from random import random
import numpy as np 


BATCH_SIZE = 8
STEPS = 3000


'''
tensor represents data
Graph constructs network
Session processes network
'''


### 1. tensor tf.Tensor

# 0 -rank tensor --> single number 
T_0 = 1

# 1- rank tensor --> vector

T_1 = [1,2,3]

# 2-rank tensor --> matrix i row , j column  M[i]][j]
T_2 = [[1,2,3],[4,5,6],[7.8,9]]

# data type
a = tf.constant([1.0,2.0]) # define a tensor equal to [1.0,2.0]
b = tf.constant([3.0,4.0])

result = a+ b

# print(result)
# Tensor("add:0", shape=(2,), dtype=float32)


### 2. Graph

x = tf.constant([[1.0,2.0]]) # 1 * 2
w = tf.constant([[3.0],[4.0]]) # 2 * 1 
y = tf.matmul(x,w)

# print(y)
# Tensor("MatMul:0", shape=(1, 1), dtype=float32)


### 3. Session 
# with tf.Session() as sess:
#     print(sess.run(y))

# [[11.]]

### 4. params --> weights


'''
# randomly generate params 
w1 = tf.random_normal() # normal distribution

w2 = tf.truncated_normal() # truncated normal distribution

w3 = tf.random_uniform() # uniform distribution

w4 = tf.zeros()
w5 = tf.ones()
w6 = tf.fill()
w6 = tf.constant()
'''

w1 = tf.Variable(tf.random_normal([2,3],stddev=2,mean=0,seed=1))
w2 = tf.Variable(tf.truncated_normal([2,3],stddev= 2, mean =0,seed = 1))
w3 = tf.random_uniform(shape = [1,1], minval = 0,maxval = 1,dtype=tf.float16,seed=1) # uniform distribution [minval,maxval) 
w4 = tf.zeros([3,2],dtype=tf.int32)
w5 = tf.ones([3,4],dtype=tf.int32)


# print(w1)    
# print(w2)    
# print(w3)   
'''
<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32_ref>
<tf.Variable 'Variable_1:0' shape=(2, 3) dtype=float32_ref>
Tensor("random_uniform:0", shape=(1, 1), dtype=float16)
'''


### 4. network construction

#1. data preprocessing X, Y 

# 32 row 2 column
data = np.array([ [random(),random()] for i in range(32)])
data_y = np.array([ [ int(i+j > 1) ]   for i,j in  data ])

print("train data")
for i in range(len(data)):
    print(data[i],end=" --> ")
    print(data_y[i])


#2. [feedforward]construct NN , then run Session

X_in = tf.placeholder(shape=[None,2],dtype=tf.float32)
Y_real = tf.placeholder(shape=[None,1],dtype= tf.float32)

W1 = tf.Variable(tf.random_normal([2,3]))
a1 = tf.matmul(X_in,W1)

W2 = tf.Variable(tf.random_normal(shape = [3,1]))
output = tf.matmul(a1,W2)


#3. [backpropagation] feed data, optimise params

# loss function  mse = sum (( y_real - y_pred)^2) / n
loss_mse = tf.reduce_mean(tf.square(Y_real-output)) # sum square error , then take mean

# BP SGD ,Momentum, Adam
# w_n+1 = W_n - learning_rate * descent(w_n)
train_step = tf.train.GradientDescentOptimizer(learning_rate= 0.001).minimize(loss_mse)

#train_step = tf.train.MomentumOptimizer(learning_rate= 0.01,momentum = 0.1).minimize(loss_mse)
#train_step = tf.train.AdadeltaOptimizer(learning_rate= 0.01).minimize(loss_mse)

#4. prediction


with tf.Session() as sess:

    # Variable initialization
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(W1))
    print(sess.run(W2))

    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={X_in: data[start:end],Y_real: data_y[start:end]})
        if i % 500 == 0:
            # test 
            total_loss = sess.run(loss_mse,feed_dict={X_in:data, Y_real:data_y})
            print("After {} training steps, loss on train data is {}.".format(i,total_loss))

    print("w1{}".format(sess.run(W1)))
    print("w2{}".format(sess.run(W2)))














