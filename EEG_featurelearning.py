from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import time
import pickle
import os
"""
This code is used to extract features from the raw EEG signals for further generation.
The raw signals has 140 = 10*14 dimensions, we can compress it to 40 dimension or fewer. 
"""

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


# this function only calculate the acc of CNN_task
def compute_accuracy(v_xs, v_ys):
    global pred
    y_pre = sess.run(pred, feed_dict={xs: v_xs,})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ground: v_ys})
    return result


t1 = time.time()
"""
Data reading, all_01.dat, the first person (shape_01.pk), [3200, 2380],
each sample has 2380 features, 2380 = 10*14 + 40*56
"""
data = np.load(open('all_01.dat', 'rb'))

EEG = preprocessing.minmax_scale(data[:, :140], axis=1)
EEG = EEG * 2 - 1
ima = preprocessing.minmax_scale(data[:, 140:], axis=1)
data = np.hstack((EEG, ima))
img_size = 40 * 56
noise_size = 140
# print(data.shape)

"""
label making
"""
l0 = np.zeros([int(3200/5), 1])
for lab in range(1, 5):
    l_ = np.ones([int(3200/5), 1]) * lab
    l0 = np.vstack((l0, l_))

# concate with the label
data = np.hstack((data, l0))
np.random.shuffle(data)

n_class = 5
label = data[:, (img_size+noise_size):(img_size+noise_size)+1]
label = one_hot(label)
feature = data[:, :140]

# batch split
batch_size = label.shape[0]/5
train_fea = []
n_group = int(label.shape[0]/batch_size)

for i in range(n_group):
    f = feature[int(0+batch_size*i):int(batch_size+batch_size*i)]
    train_fea.append(f)
# print(train_fea[0].shape)

train_label=[]
for i in range(n_group):
    f = label[int(0 + batch_size * i):int(batch_size + batch_size * i), :]
    train_label.append(f)

# print(train_label[0].shape)
test_fea = train_fea[-1]
test_label = train_label[-1]

"""CNN classifier"""

xs = tf.placeholder(tf.float32, shape=[None, 140], name='input_EEG')
ground = tf.placeholder(tf.float32, shape=[None, n_class], name='ground_truth')

z_image = tf.reshape(xs, [-1, 10, 14, 1])
depth_1 = 32
depth_2 = 64
out_dim = 40
conv1 = tf.layers.conv2d(inputs=z_image, filters=depth_1, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])

conv2 = tf.layers.conv2d(inputs=pool1, filters=depth_2, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])

flatten = tf.contrib.layers.flatten(pool2)
flatten = tf.nn.dropout(flatten, 0.5)

fea = tf.layers.dense(flatten, out_dim, activation=tf.nn.sigmoid)

pred = tf.layers.dense(fea, n_class)

# cost and accuracy
l2 = 0.0005 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ground))+l2

train_step = tf.train.AdamOptimizer(0.0005).minimize(cost)

# use this to limit the GPU number

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

init = tf.global_variables_initializer()
sess.run(init)

step = 1
while step < 1000:
    for i in range(n_group-1):
        feed = feed_dict = {xs:train_fea[i], ground:train_label[i]}
        sess.run(train_step, feed_dict=feed)
    if step % 100 == 0:
        """ cost"""
        cost_train =sess.run(cost, feed_dict=feed)
        cost_test =sess.run(cost, feed_dict={xs:test_fea, ground:test_label})
        acc_test = compute_accuracy(test_fea, test_label)

        print('the step is:', step, 'train, test, acc task', compute_accuracy(train_fea[0], train_label[0]), acc_test
                , ', train test cost', cost_train, cost_test)
        if acc_test > 0.6:
            break

    step += 1
#
EEG_features = sess.run(fea, feed_dict={xs : data[:, :140]})
all_data = np.hstack((EEG_features, data[:, 140:]))

pickle.dump(all_data, open('shape_EEG_feature.pkl', 'wb'))
print('dumped as shape_EEG_feature.pkl', data.shape, all_data.shape, time.time() - t1)



