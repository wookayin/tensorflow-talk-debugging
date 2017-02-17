import tensorflow as tf
import tensorflow.contrib.layers as layers
from datetime import datetime

# MNIST input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    tf.summary.histogram('fc1', fc1)
    tf.summary.histogram('fc1/sparsity', tf.nn.zero_fraction(fc1))
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    tf.summary.histogram('fc2', fc2)
    tf.summary.histogram('fc2/sparsity', tf.nn.zero_fraction(fc2))
    out = layers.fully_connected(fc2, 10, activation_fn=None, scope='out')
    return out

# build model, loss, and train op
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
pred = multilayer_perceptron(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, dtype=tf.int32, trainable=False)   # (*)
train_op = tf.train.AdamOptimizer(learning_rate=0.001)\
                   .minimize(loss, global_step=global_step)     # (*)

# histogram summary for all trainable variables (slow?)
for v in tf.trainable_variables():
    tf.summary.histogram(v.name, v)

def train(session):
    batch_size = 200
    session.run(tf.global_variables_initializer())
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/tmp/mnist', session.graph)

    # Training cycle
    for epoch in range(10):
        epoch_loss = 0.0
        batch_steps = mnist.train.num_examples / batch_size
        for step in range(batch_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c, summary = session.run(
                [train_op, loss, merged_summary_op],
                feed_dict={x: batch_x, y: batch_y})
            summary_writer.add_summary(summary, global_step.eval(session=session))
            epoch_loss += c / batch_steps
        print "[%s] Epoch %02d, Loss = %.6f" % (datetime.now(), epoch, epoch_loss)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

def main():
    with tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        device_count={'GPU': 1})) as session:
        train(session)

if __name__ == '__main__':
    main()
