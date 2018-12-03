# 神经网路 优化

#f(  \sum{w_i*x_i} + b )  --> activation function


'''
tf.nn.relu
tf.nn.sigmond
tf.nn.tanh
'''

########## 1. loss function 
# mse, ce 

import tensorflow as tf 
import numpy as np 


BATCH_SIZE = 8
SEED = 2345

# numpy.random.RandomState.rand(d0,d1,d2.....dn)  [0,1)
rds = np.random.RandomState(SEED)
trainSetX = rds.rand(32,2)
trainSetY = [ [ 2*a-3*b + rds.rand()/10.0-0.05 ] for a,b in trainSetX ]


print("\n\n\n### training data X-->Y ")
for i in range(len(trainSetX)):
    print(trainSetX[i],end= " --> ")
    print(trainSetY[i])

# graph 
X = tf.placeholder(shape=[None,2], dtype = tf.float32)
Y = tf.placeholder(shape=[None,1],dtype = tf.float32)

W1 = tf.Variable(tf.random_normal(shape = [2,1]))
y_ = tf.matmul(X,W1)

# loss function --  mse
loss = tf.reduce_mean(tf.square(Y-y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
'''
 after 0 steps training, loss is 3.8073272705078125
W1 [[-0.33184704]
 [ 1.8394799 ]]
 after 500 steps training, loss is 0.3820086717605591
W1 [[ 0.6058756]
 [-1.4624807]]
 after 1000 steps training, loss is 0.061306655406951904
W1 [[ 1.4278007]
 [-2.3711212]]
 after 1500 steps training, loss is 0.010351511649787426
W1 [[ 1.756508]
 [-2.732346]]
 after 2000 steps training, loss is 0.0022557766642421484
W1 [[ 1.887539]
 [-2.876334]]
 after 2500 steps training, loss is 0.000969726825132966
W1 [[ 1.9397703]
 [-2.9337292]]
'''

# train_step = tf.train.AdamOptimizer().minimize(loss)

'''
 after 0 steps training, loss is 0.8843720555305481
W1 [[ 0.28786308]
 [-0.5367955 ]]
 after 500 steps training, loss is 0.600144624710083
W1 [[ 0.44186625]
 [-0.984892  ]]
 after 1000 steps training, loss is 0.393183171749115
W1 [[ 0.6931101]
 [-1.3769237]]
 after 1500 steps training, loss is 0.242482990026474
W1 [[ 0.9523464]
 [-1.7267144]]
 after 2000 steps training, loss is 0.13818895816802979
W1 [[ 1.1960344]
 [-2.0365155]]
 after 2500 steps training, loss is 0.07103285193443298
W1 [[ 1.4140239]
 [-2.3048012]]
 after 3000 steps training, loss is 0.0319242924451828
W1 [[ 1.5994151]
 [-2.528389 ]]
 after 3500 steps training, loss is 0.012151947245001793
W1 [[ 1.746787 ]
 [-2.7038321]]
 after 4000 steps training, loss is 0.003955007065087557
W1 [[ 1.8531624]
 [-2.8294828]]
 after 4500 steps training, loss is 0.0013716125395148993
W1 [[ 1.9201185]
 [-2.9082344]]
'''
# BP 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    STEPS =5000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={X:trainSetX[start:end], Y:trainSetY[start:end]})
        # test 
        if i % 500 == 0:
            total_loss = sess.run(loss,feed_dict={X:trainSetX,Y:trainSetY})
            print(" after {} steps training, loss is {}".format(i,total_loss))
            print("W1 {}".format(sess.run(W1)))

    

# cross entropy


# ce = -tf.reduce_mean(y_* tf.log(tf.clip_by_value(y,1e-12,1.0)))

# ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_,1))
# cem = tf.reduce_mean(ce)

######## 2. learning rate 每次更新的幅度

# wn+1 = wn - learning_rate * descent(W)

########## 3. 滑动平均


########## 4. 正则化 regularization



