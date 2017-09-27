import numpy as np 
import tensorflow as tf
from numpy import genfromtxt
sess = tf.InteractiveSession()


my_data = genfromtxt('/home/nitish/Downloads/train.csv', delimiter=',')
test_data= genfromtxt('/home/nitish/Downloads/test.csv', delimiter=',')

def one_hot(i):
	a = np.zeros((i.size,10))
	for j in range(i.size):
		a[j,int(i[j])]=1
	return a        

test_data = np.delete(test_data,0,0)
input_data = np.delete(my_data,0,0)
train_labels = np.transpose(input_data[:,0])
input_data = np.delete(input_data,0,axis=1)
train = input_data
test = test_data
train_labels = one_hot(train_labels)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean (tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(6000):
	if i%420==0:
	  (train,train_labels) = unison_shuffled_copies(train, train_labels)
	batch_0 = train[((100*i)%42000) : ((100*(i+1))%42000),:]
	batch_1 = train_labels[((100*i)%42000): ((100*(i+1))%42000),:]
	if i%100 == 0:
	  train_accuracy = accuracy.eval(feed_dict={
		x:batch_0, y_: batch_1, keep_prob: 1.0})
	  print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch_0, y_: batch_1, keep_prob: 0.5})
	

test_trial = tf.nn.softmax(y_conv)
test_trial = test_trial.eval(feed_dict={x: test[0:5000,:], keep_prob: 1})
test_trial = np.array(list(test_trial))
test_label = np.argmax(test_trial, axis=1	)

np.savetxt('/home/nitish/Desktop/Trial_1.csv', test_label, delimiter="," )

test_trial = tf.nn.softmax(y_conv)
test_trial = test_trial.eval(feed_dict={x: test[5000:11000,:], keep_prob: 1})
test_trial = np.array(list(test_trial))
test_label = np.argmax(test_trial, axis=1	)

np.savetxt('/home/nitish/Desktop/Trial_2.csv', test_label, delimiter="," )

test_trial = tf.nn.softmax(y_conv)
test_trial = test_trial.eval(feed_dict={x: test[11000:17000,:], keep_prob: 1})
test_trial = np.array(list(test_trial))
test_label = np.argmax(test_trial, axis=1	)

np.savetxt('/home/nitish/Desktop/Trial_3.csv', test_label, delimiter="," )

test_trial = tf.nn.softmax(y_conv)
test_trial = test_trial.eval(feed_dict={x: test[17000:23000,:], keep_prob: 1})
test_trial = np.array(list(test_trial))
test_label = np.argmax(test_trial, axis=1	)

np.savetxt('/home/nitish/Desktop/Trial_4.csv', test_label, delimiter="," )

test_trial = tf.nn.softmax(y_conv)
test_trial = test_trial.eval(feed_dict={x: test[23000:28000,:], keep_prob: 1})
test_trial = np.array(list(test_trial))
test_label = np.argmax(test_trial, axis=1	)

np.savetxt('/home/nitish/Desktop/Trial_5.csv', test_label, delimiter="," )