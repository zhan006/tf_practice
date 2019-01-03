import tensorflow as tf
from mnist import MNIST
import mnist
import matplotlib.pyplot as plt
data=MNIST()
imgshape=data.img_shape
weights=tf.Variable(tf.zeros([784,10]))
bias=tf.Variable(tf.zeros([10]))
x=tf.placeholder('float',[None,784])
logit=tf.matmul(x,weights)+bias
y_pred=tf.nn.softmax(logit)
y_pred_cls=tf.argmax(y_pred,1)
y_=tf.placeholder('float',[None,10])
loss = -tf.reduce_mean(y_ * tf.log(y_pred))
x_test=data.x_test
y_test=data.y_test
y_cls=data.y_test_cls
#print(y_cls)
sess=tf.Session()
optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
sess.run(tf.global_variables_initializer())
y_true=tf.placeholder(tf.int64,[None])
accuracy=tf.equal(y_true,y_pred_cls)
acc=tf.reduce_mean(tf.cast(accuracy,tf.float32))
def plotnumber(arr,label,pre_label=None):
  fig,axe=plt.subplots(3,3)
  fig.subplots_adjust(hspace=1,wspace=0.3)
  for i,ax in enumerate(axe.flat):
    ax.imshow(arr[i].reshape(img_shape))
    
    ax.set_xticks([])
    ax.set_yticks([])
    if pre_label is None:
        xlabel=label[i]
        ax.set_xlabel('true: '+str(xlabel))
    else:
        xlabel=label[i]
        tlabel=pre_label[i]
        ax.set_xlabel('true:{} pre:{}'.format(xlabel,tlabel))
  
  plt.show()


  
def training(number):
    for i in range(number):
        xbat,ybat,_=data.random_batch(100)
        feeddict={x:xbat,y_:ybat}
        sess.run(optimizer,feed_dict=feeddict)                  
def acce():
    feed={x:x_test,y_true:y_cls}
    a=sess.run(acc,feed_dict=feed)
    print(a)
def prediction():
  feed={x:x_test}
  pr=sess.run(y_pred_cls,feed_dict=feed)
  return pr
training(1000)
acce()
pr=prediction()
plotnumber(x_test[0:9],y_cls[0:9],pr[0:9])

