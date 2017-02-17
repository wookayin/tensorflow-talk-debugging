import tensorflow as tf
import tensorflow.contrib.layers as layers
from datetime import datetime

# MNIST input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def multilayer_perceptron(x):
    global fc1, fc2
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, 10, activation_fn=tf.nn.relu, scope='out') #

    def _debug_func(x_val, fc1_val, fc2_val, out_val):
        if (out_val == 0.0).any():
            import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        #from IPython import embed; embed()  # XXX DEBUG
        return False

    debug_op = tf.py_func(_debug_func, [x, fc1, fc2, out], [tf.bool])
    with tf.control_dependencies(debug_op):
        out = tf.identity(out, name='out')
    return out

# build model, loss, and train op
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
pred = multilayer_perceptron(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

def train(session):
    batch_size = 200
    session.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(10):
        epoch_loss = 0.0
        batch_steps = mnist.train.num_examples / batch_size
        for i in range(batch_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = session.run([train_op, loss],
                               feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c / batch_steps
        print "[%s] Epoch %02d, Loss = %.6f" % (datetime.now(), epoch, epoch_loss)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

def main():
    global session
    with tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        device_count={'GPU': 1})) as session:
        train(session)

if __name__ == '__main__':
    main()
