import tensorflow as tf
from tensorflow.keras import datasets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    """
    :param x:
    :param y:
    :return:
    """
    # [b, 28, 28], [b]
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x, y


(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(60000).batch(128).map(preprocess).repeat(30)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(10000).batch(128).map(preprocess)
x,y = next(iter(train_db))
print('train sample:', x.shape, y.shape)
print(x[0], y[0])


def main():

    # learning rate
    lr = 1e-3

    # w1: (784 x 512); b1: (1 x 512)
    w1, b1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1)), tf.Variable(tf.zeros([512]))
    # w2: (512 x 256); b2: (1 x 256)
    w2, b2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # w3: (256 x 10); b2: (1 x 10)
    w3, b3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    for step, (x,y) in enumerate(train_db):

        # [128, 28, 28] => [128, 784]
        x = tf.reshape(x, (-1, 784))

        with tf.GradientTape() as tape:

            # Neural network
            # layer1.
            # (128 x 784) @ (784 x 512) + (1 x 512) = (128 x 512)
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)

            # layer2
            # (128 x 512) @ (512 x 256) + (1 x 256) = (128 x 256)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            # output
            # (128 x 256) @ (256 x 10) + (1 x 10) = (128 x 10)
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # compute loss
            # [128, 10] - [128, 10]
            loss = tf.square(y-out)
            # [128, 10] => [128]
            loss = tf.reduce_mean(loss, axis=1)
            # [128] => scalar
            loss = tf.reduce_mean(loss)

        # compute gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # for g in grads:
        #     print(tf.norm(g))
        # update w' = w - lr*grad
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)

        # print loss
        if step % 100 == 0:
            print(step, 'loss:', float(loss))

        # evaluate
        if step % 500 == 0:
            total, total_correct = 0., 0

            for step, (x, y) in enumerate(test_db):
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b] predicted value
                pred = tf.argmax(out, axis=1)
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)
                # bool type [b]
                correct = tf.equal(pred, y)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct/total)


if __name__ == '__main__':
    main()
