import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#pd.set_option("max_columns",10)

# 读取数据
train_data = pd.read_csv("data/train.csv",sep =",")[1000:]
test_data = pd.read_csv("data/test.csv",sep = ",")[100:]
tr_data_x, ts_data_x,tr_data_y,ts_data_y = train_test_split(train_data.iloc[:,1:],train_data['label'],test_size=0.1)
#print(tr_data_x.shape,tr_data_y.shape,ts_data_x.shape,ts_data_y.shape)

tr_data_x = tr_data_x.values.reshape([-1,28,28,1])
ts_data_x = ts_data_x.values.reshape([-1,28,28,1])
#print(tr_data_x.shape,ts_data_x.shape)
# onehotencoder
enc = OneHotEncoder()
enc.fit([[i] for i in range(10)])
tr_data_y  = enc.transform(tr_data_y.values.reshape(-1,1)).toarray()
ts_data_y = enc.transform(ts_data_y.values.reshape(-1,1)).toarray()


# 构建CNN 网络架构
X = tf.placeholder("float",[None,28,28,1])
Y = tf.placeholder("float",[None,10])
keep_conv = tf.placeholder("float")
keep_hidden = tf.placeholder("float")
# 初始化权重W 
def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.01))
# init bias
# def init_bias(shape):
#     return tf.constant(0.,shape = shape)



##### first layer
## convolution
w1 = init_weights([3,3,1,32])
#b1 = init_bias([32])
l1 = tf.nn.conv2d(X,w1,[1,1,1,1],padding="SAME") 
l1_act = tf.nn.relu(l1)
#print(l1_act)
## pooling
l1_act_pool = tf.nn.max_pool(l1_act,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#print(l1_act_pool)
## dropout
l1_act_pool_dropout = tf.nn.dropout(l1_act_pool,keep_conv)

##### second layer
## convolution
w2 = init_weights([3,3,32,64])
#b2 = init_bias([64])
l2 = tf.nn.conv2d(l1_act_pool_dropout,w2,[1,1,1,1],padding="SAME") 
l2_act = tf.nn.relu(l2)
#print(l2_act)
## pooling
l2_act_pool = tf.nn.max_pool(l2_act,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#print(l2_act_pool)
## dropout
l2_act_pool_dropout = tf.nn.dropout(l2_act_pool,keep_conv)

##### third layer
## convolution
w3 = init_weights([3,3,64,128])
#b3 = init_bias([128])
l3 = tf.nn.conv2d(l2_act_pool_dropout,w3,[1,1,1,1],padding="SAME") 
l3_act = tf.nn.relu(l3)
#print(l3_act)
## pooling
l3_act_pool = tf.nn.max_pool(l3_act,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#print(l3_act_pool)
## dropout
l3_act_pool_dropout = tf.nn.dropout(l3_act_pool,keep_conv)

##### fc layer 
w4 = init_weights([128 * 4 * 4, 625])
#b4  = init_bias([625])
l4 = tf.matmul(tf.reshape(l3_act_pool_dropout,[-1,w4.get_shape().as_list()[0]]),w4) 
l4_act = tf.nn.relu(l4)
l4_act_dropout = tf.nn.dropout(l4_act,keep_hidden)


### output layer
w_o = init_weights([l4_act_dropout.get_shape().as_list()[-1],10])
#b_o = init_bias([10])
l_o = tf.matmul(l4_act_dropout,w_o)


### loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=l_o,labels=Y))
train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
predict_op = tf.argmax(l_o,axis=1)

# defination
batch = 128
test_size = 256
### training
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(20):
        training_batch = zip(range(0,len(tr_data_x),batch),range(batch,len(tr_data_x)+1,batch))
        
        for start ,end in training_batch:
            print(start,end)
            sess.run(train_op,feed_dict={X:tr_data_x,Y:tr_data_y,keep_conv :0.8,keep_hidden :0.5 })

        # 随机采样test_size 个 测试样例
        test_indices = np.arange(len(ts_data_x))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        print('{}test:'.format(i))

        print(i,np.mean(np.argmax(ts_data_y[test_indices],axis=1) == \
                        sess.run(predict_op,feed_dict={X:ts_data_x[test_indices],keep_conv : 1.0,keep_hidden: 1.0})))