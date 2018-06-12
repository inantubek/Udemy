import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta

import b_b_cifar10 as cifar10

cifar10.download()

print(cifar10.load_class_names())

train_img, train_cls, train_labels = cifar10.load_training_data()
test_img, test_cls, test_labels = cifar10.load_test_data()

print('Training set:', len(train_img), 'Testing set:', len(test_img))

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, [None, 10])
pkeep = tf.placeholder(tf.float32)

def pre_process_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)

    return image

def pre_process(images):
    images = tf.map_fn(lambda image: pre_process_image(image), images)
    return images

distorted_images = pre_process(images=x)

# gpu üzerinde pre_process yavaş olduğunda pre_process işlemini cpu ya veriyoruz
"""
with tf.device("/cpu:0"):
    distorted_images = pre_process(images=x)
"""

def conv_layer(input, size_in, size_out, use_pooling=True):
    w = tf.Variable(tf.truncated_normal([3, 3, size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))

    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME") + b
    y = tf.nn.relu(conv)

    if use_pooling:
        y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    return y

def fc_layer(input, size_in, size_out, relu=True, dropout=True):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    logits = tf.matmul(input, w) + b

    if relu:
        y = tf.nn.relu(logits)
        if dropout:
            y = tf.nn.dropout(y, pkeep)
        return y
    else:
        return logits

conv1 = conv_layer(distorted_images, 3, 32, use_pooling=True)
conv2 = conv_layer(conv1, 32, 64, use_pooling=True)
conv3 = conv_layer(conv2, 64, 64, use_pooling=True)

flattened = tf.reshape(conv3, [-1, 4 * 4 * 64])
fc1 = fc_layer(flattened, 4 * 4 * 64, 512, relu=True, dropout=True)
fc2 = fc_layer(fc1, 512, 256, relu=True, dropout=True)
logits = fc_layer(fc2, 256, 10, relu=False, dropout=False)
y = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(y, 1)

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128

def random_batch():
    index = np.random.choice(len(train_img), size=batch_size, replace=False)
    x_batch = train_img[index, :, :, :]
    y_batch = train_labels[index, :]

    return x_batch, y_batch

loss_graph = []

def training_step(iterations):
    start_time = time.time()
    for i in range(iterations):
        x_batch, y_batch = random_batch()
        feed_dict_train = {x: x_batch, y_true: y_batch, pkeep: 0.5}
        [_, train_loss] = sess.run([optimizer, loss], feed_dict=feed_dict_train)
        loss_graph.append(train_loss)

        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print("Iteration:", i, "Training Accuracy:", acc, "Training Loss:", train_loss)

    end_time = time.time()
    time_diff = end_time - start_time
    print("Time usage: ", timedelta(seconds=int(round(time_diff))))

batch_size_test = 256

def test_accuracy():
    num_images = len(test_img)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0
    while i < num_images:
        j = min(i + batch_size_test, num_images)
        feed_dict = {x: test_img[i:j, :], y_true: test_labels[i:j, :], pkeep: 1}
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    correct = (test_cls == cls_pred)
    print("Testing Accuracy: ", correct.mean())

def plot_images(images, cls_true, smooth=True):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)

    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    class_names = cifar10.load_class_names()

    for i, ax in enumerate(axes.flat):
        if smooth:
            interpolation = "spline16"
        else:
            interpolation = "nearest"

        ax.imshow(images[i, :, :, :], interpolation=interpolation)
        cls_true_name = class_names[cls_true[i]]

        xlabel = "True: {}".format(cls_true_name)

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def distorted_image(image, cls_true):
    image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)
    feed_dict = {x: image_duplicates}
    result = sess.run(distorted_images, feed_dict=feed_dict)
    plot_images(images=result, cls_true=np.repeat(cls_true, 9))

def plot_distorted_image(i):
    return distorted_image(test_img[i, :, :, :], test_cls[i])

# plot_distorted_image(8)

training_step(1000)
test_accuracy()

plt.plot(loss_graph, 'k-')
plt.title("Loss Grafiği")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
