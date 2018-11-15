#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

rnn_unit=25       #隐层数量
input_size=8
output_size=1
lr=0.0003         #学习率
time_step=5
batch_size=20
lstm_layers=2

#——————————————————导入数据——————————————————————
data = pd.read_csv("2330_v2.csv", header=0, sep='\t', 
                    lineterminator='\r', encoding = 'utf-8')
data = data.dropna()
data_num = data.shape[0]

#data[:,8] = data[:,8].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))
#remove comma
data.iloc[:,1] = data.iloc[:,1].str.replace(',','')
data.iloc[:,2] = data.iloc[:,2].str.replace(',','')
data.iloc[:,8] = data.iloc[:,8].str.replace(',','')

#convert datatype:object to numeric type
data = data.convert_objects(convert_numeric=True)

#generate label data: use the previous date's close price
data['label'] = data['close']

for i in range(0, data_num):
    if i < data_num -1:
        data.loc[i, 'label'] = data.loc[i+1, 'close']
    else:
        data.loc[i, 'label'] = data.loc[i, 'close']

# transform pandas to numpy
data=data.iloc[:,1:10].values  #取第2-10列


#获取训练集
def get_train_data(batch_size=batch_size,time_step=time_step,train_begin=0,train_end=data_num-1):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:8]
       y=normalized_train_data[i:i+time_step,8,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y


#获取测试集
def get_test_data(time_step=time_step,test_begin=data_num-30):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:8]
       y=normalized_test_data[i*time_step:(i+1)*time_step,8]
       test_x.append(x.tolist())
       test_y.extend(y)
       #test_x.append((normalized_test_data[(i+1)*time_step:,:8]).tolist())
       #test_y.extend((normalized_test_data[(i+1)*time_step:,8]).tolist())
    return mean,std,test_x,test_y


#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

#——————————————————定义神经网络变量——————————————————
def lstmCell():
    #basicLstm单元
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm

def lstm(X):
    
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell=tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32,
                                              swap_memory=True)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#————————————————训练模型————————————————————

def train_lstm(batch_size=batch_size,time_step=time_step,train_begin=0,train_end=data_num-1):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(250):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print("Number of iterations:",i," loss:",loss_)
        print("model_save: ",saver.save(sess,'model_save2/modle.ckpt'))
        #我是在window下跑的，这个地址是存放模型的地方，模型参数文件名为modle.ckpt
        #在Linux下面用 'model_save2/modle.ckpt'
        print("The train has finished")
train_lstm()

#————————————————预测模型————————————————————
def prediction(time_step=time_step):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    with tf.variable_scope("sec_lstm",reuse=tf.AUTO_REUSE):
        pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[8]+mean[8]
        test_predict=np.array(test_predict)*std[8]+mean[8]
        print("test_predict:", test_predict)
        print("test_y:", test_y)

        from sklearn.metrics import r2_score
        from sklearn.metrics import mean_squared_error

        print('MSE train: %.3f' % (
        mean_squared_error(test_y, test_predict)))

        print('R^2 train: %.3f' % (
        r2_score(test_y, test_predict)))

        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差程度
        print("The accuracy of this predict:", acc)
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b',)
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

prediction()
