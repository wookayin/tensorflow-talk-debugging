name: inverse
class: center, middle, inverse
layout: true

---

class: titlepage, no-number

# A Practical Guide for Debugging TensorFlow Codes

## .author[Jongwook Choi]

### .small[.white[Feb 17th, 2017] <br/> .green[Initial Version: June 18th, 2016]]

### .x-small[https://github.com/wookayin/tensorflow-talk-debugging]


---

layout: false

## Bio: Jongwook Choi ([@wookayin][wookayin-gh])

* An undergraduate student from Seoul National University, <br/> [Vision and Learning Laboratory][snuvl-web]
* Looking for a graduate (Ph.D) program in ML/DL
* A huge fan of TensorFlow and Deep Learning 😀
  <br/> .dogdrip[TensorFlow rocks!!!!]

.right.img-33[![Profile image](images/profile-wook.png)]


[snuvl-web]: http://vision.snu.ac.kr
[wookayin-gh]: https://github.com/wookayin


---

## About

This talk aims to share you with some practical guides and tips on writing and debugging TensorFlow codes.

--

... because you might find that debugging TensorFlow codes is something like ...


---
class: center, middle, no-number, bg-full
background-image: url(images/meme-doesnt-work.jpg)
background-repeat: no-repeat
background-size: contain

---
## Welcome!

### .green[Contents]

- Introduction: Why debugging in TensorFlow is difficult
- Basic and advanced methods for debugging TensorFlow codes
- General tips and guidelines for easy-debuggable code
- .dogdrip[Benchmarking and profiling TensorFlow codes]


---

### .red[A Disclaimer]

- This talk is NOT about how to debug your *ML/DL model*
  .gray[(e.g. my model is not fitting well)],
  but about how to debug your TF codes *in a programming perspective*
- I had to assume that the audience is somewhat familiar with basics of TensorFlow and Deep Learning;
  it would be very good if you have an experience to write a TensorFlow code by yourself
- Questions are highly welcomed! Please feel free to interrupt!


---

template: inverse

# Debugging?

---

## Debugging TensorFlow Application is ...

--

- Difficult!

--

- Do you agree? .dogdrip[Life is not that easy]

---

## Review: TensorFlow Computation Mechanics

* The core concept of TensorFlow: **The Computation Graph**
* See Also: [TensorFlow Mechanics 101](https://www.tensorflow.org/versions/master/tutorials/mnist/tf/index.html)

.center.img-33[![](images/tensors_flowing.gif)]


---

## Review: TensorFlow Computation Mechanics

.gray.right[(from [TensorFlow docs](https://www.tensorflow.org/versions/master/get_started/basic_usage.html))]

TensorFlow programs are usually structured into
- a .green[**construction phase**], that assembles a graph, and
- an .blue[**execution phase**] that uses a session to execute ops in the graph.

.center.img-33[![](images/tensors_flowing.gif)]

---

## Review: in pure numpy ...

```python
W1, b1, W2, b2, W3, b3 = init_parameters()

def multilayer_perceptron(x, y_truth):
    # (i) feed-forward pass
    assert x.dtype == np.float32 and x.shape == [batch_size, 784] # numpy!
*   fc1 = fully_connected(x, W1, b1, activation_fn='relu')    # [B, 256]
*   fc2 = fully_connected(fc1, W2, b2, activation_fn='relu')  # [B, 256]
    out = fully_connected(fc2, W3, b3, activation_fn=None)    # [B, 10]

    # (ii) loss and gradient, backpropagation
    loss = softmax_cross_entropy_loss(out, y_truth)   # loss as a scalar
    param_gradients = _compute_gradient(...)  # just an artificial example :)
    return out, loss, param_gradients

def train():
    for epoch in range(10):
        epoch_loss = 0.0
        batch_steps = mnist.train.num_examples / batch_size
        for step in range(batch_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
*           y_pred, loss, gradients = multilayer_perceptron(batch_x, batch_y)
            for v, grad_v in zip(all_params, gradients):
                v = v - learning_rate * grad_v
            epoch_loss += c / batch_steps
        print "Epoch %02d, Loss = %.6f" % (epoch, epoch_loss)
```

---

## Review: with TensorFlow


```python
def multilayer_perceptron(x):
*   fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
*   fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu)
    out = layers.fully_connected(fc2, 10, activation_fn=None)
    return out

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
*pred = multilayer_perceptron(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

def train(session):
    batch_size = 200
    session.run(tf.initialize_all_variables())

    for epoch in range(10):
        epoch_loss = 0.0
        batch_steps = mnist.train.num_examples / batch_size
        for step in range(batch_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
*           _, c = session.run([train_op, loss], {x: batch_x, y: batch_y})
            epoch_loss += c / batch_steps
        print "Epoch %02d, Loss = %.6f" % (epoch, epoch_loss)
```

---

## Review: The Issues

```python
def multilayer_perceptron(x):
*   fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
*   fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu)
    out = layers.fully_connected(fc2, 10, activation_fn=None)
    return out

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
*pred = multilayer_perceptron(x)
```

- The actual computation is done inside `session.run()`;
  what we just have done is to build a computation graph
- The model building part (e.g. `multilayer_perceptron()`) is called only *once*
  before training, so we can't access the intermediates simply
  - e.g. Inspecting activations of `fc1`/`fc2` is not trivial!

```python
*           _, c = session.run([train_op, loss], {x: batch_x, y: batch_y})
```

---

## [`Session.run()`][apidocs-sessionrun]

The most important method in TensorFlow --- where every computation is performed!

- `tf.Session.run(fetches, feed_dict)` runs the operations and evaluates in `fetches`,
  subsituting the values (placeholders) in `feed_dict` for the corresponding input values.

[apidocs-sessionrun]: https://www.tensorflow.org/versions/master/api_docs/python/train.html#scalar_summary


---

## Why TensorFlow debugging is difficult?

- *The concept of Computation Graph* might be unfamiliar to us.
- The "Inversion of Control"
    - The actual computation (feed-forward, training) of model runs inside `Session.run()`,
      upon the computation graph, **but not upon the Python code we wrote**
    - What is exactly being done during an execution of session is under an abstraction barrier
- Therefore, we do not retrieve the intermediate values during the computation,
  unless we explicitly fetch them via `Session.run()`


<!-- ============================================================================================ -->
<!-- ============================================================================================ -->

---

template: inverse

# Debugging Facilities in TensorFlow


---

## Debugging Scenarios

We may wish to ...

- inspect intra-layer .blue[activations] (during training)
  - e.g. See the output of conv5, fc7 in CNNs
- inspect .blue[parameter weights] (during training)
- under some conditions, pause the execution (i.e. .blue[breakpoint]) and
  evaluate some expressions for debugging
- during training, .red[*NaN*] occurs in loss and variables .dogdrip[but I don't know why]


--

😀  Of course, in TensorFlow, we can do these very elegantly!

---

## Debugging in TensorFlow: Overview

.blue[**Basic ways:**]

* Explicitly fetch, and print (or do whatever you want)!
  * `Session.run()`
* Tensorboard: Histogram and Image Summary
* the `tf.Print()` operation

.blue[**Advanced ways:**]

* Interpose any python codelet in the computation graph
* A step-by-step debugger
* `tfdbg`: The TensorFlow debugger


---

## (1) Fetch tensors via `Session.run()`

TensorFlow allows us to run parts of graph in isolation, i.e.
.green[only the relevant part] of graph is executed (rather than executing *everything*)

```python
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
bias = tf.Variable(1.0)

y_pred = x ** 2 + bias     # x -> x^2 + bias
loss = (y - y_pred)**2     # l2 loss?

# Error: to compute loss, y is required as a dependency
print('Loss(x,y) = %.3f' % session.run(loss, {x: 3.0}))

# OK, print 1.000 = (3**2 + 1 - 9)**2
print('Loss(x,y) = %.3f' % session.run(loss, {x: 3.0, y: 9.0}))

# OK, print 10.000; for evaluating y_pred only, input to y is not required
*print('pred_y(x) = %.3f' % session.run(y_pred, {x: 3.0}))

# OK, print 1.000 bias evaluates to 1.0
*print('bias      = %.3f' % session.run(bias))
```

---

## Tensor Fetching: Example

We need to access to the tensors as python expressions

```python
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu)
    out = layers.fully_connected(fc2, 10, activation_fn=None)
*   return out, fc1, fc2

net = {}
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
*pred, net['fc1'], net['fc2'] = multilayer_perceptron(x)
```

to fetch and evaluate them:
```python
            _, c, fc1, fc2, out = session.run(
*               [train_op, loss, net['fc1'], net['fc2'], pred],
                feed_dict={x: batch_x, y: batch_y})

            # and do something ...
            if step == 0: # XXX Debug
                print fc1[0].mean(), fc2[0].mean(), out[0]
```


---

## (1) Fetch tensors via `Session.run()`

.green[**The Good:**]

* Simple and Easy.
* The most basic method to get debugging information.
* We can fetch any evaluation result in numpy arrays, .green[anywhere] .gray[(except inside `Session.run()` or the computation graph)].


--

.red[**The Bad:**]

* We need to hold the reference to the tensors to inspect,
  which might be burdensome if model becomes complex and big
  <br> .gray[(Or, we can simply pass the tensor name such as `fc0/Relu:0`)]
* The feed-forward needs to be done in an atomic way (i.e. a single call of `Session.run()`)


---

## Tensor Fetching: The Bad (i)

* We need to hold a reference to the tensors to inspect,
  which might be burdensome if model becomes complex and big

```python
def alexnet(x):
    assert x.get_shape().as_list() == [224, 224, 3]
    conv1 = conv_2d(x, 96, 11, strides=4, activation='relu')
    pool1 = max_pool_2d(conv1, 3, strides=2)
    conv2 = conv_2d(pool1, 256, 5, activation='relu')
    pool2 = max_pool_2d(conv2, 3, strides=2)
    conv3 = conv_2d(pool2, 384, 3, activation='relu')
    conv4 = conv_2d(conv3, 384, 3, activation='relu')
    conv5 = conv_2d(conv4, 256, 3, activation='relu')
    pool5 = max_pool_2d(conv5, 3, strides=2)
    fc6 = fully_connected(pool5, 4096, activation='relu')
    fc7 = fully_connected(fc6, 4096, activation='relu')
    output = fully_connected(fc7, 1000, activation='softmax')
    return conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool5, fc6, fc7

conv1, conv2, conv3, conv4, conv5, fc6, fc7, output = alexnet(images)  # ?!
```
```python
_, loss_, conv1_, conv2_, conv3_, conv4_, conv5_, fc6_, fc7_ = session.run(
        [train_op, loss, conv1, conv2, conv3, conv4, conv5, fc6, fc7],
        feed_dict = {...})
```

---

## Tensor Fetching: The Bad (i)

* Suggestion: Using a `dict` or class instance (e.g. `self.conv5`) is a very good idea

```python
def alexnet(x, net={}):
    assert x.get_shape().as_list() == [224, 224, 3]
    net['conv1'] = conv_2d(x, 96, 11, strides=4, activation='relu')
    net['pool1'] = max_pool_2d(net['conv1'], 3, strides=2)
    net['conv2'] = conv_2d(net['pool1'], 256, 5, activation='relu')
    net['pool2'] = max_pool_2d(net['conv2'], 3, strides=2)
    net['conv3'] = conv_2d(net['pool2'], 384, 3, activation='relu')
    net['conv4'] = conv_2d(net['conv3'], 384, 3, activation='relu')
    net['conv5'] = conv_2d(net['conv4'], 256, 3, activation='relu')
    net['pool5'] = max_pool_2d(net['conv5'], 3, strides=2)
    net['fc6'] = fully_connected(net['pool5'], 4096, activation='relu')
    net['fc7'] = fully_connected(net['fc6'], 4096, activation='relu')
    net['output'] = fully_connected(net['fc7'], 1000, activation='softmax')
    return net['output']

net = {}
output = alexnet(images, net)
# access intermediate layers like net['conv5'], net['fc7'], etc.
```


---

## Tensor Fetching: The Bad (i)

* Suggestion: Using a `dict` or class instance (e.g. `self.conv5`) is a very good idea

```python
class AlexNetModel():
    # ...
    def build_model(self, x):
        assert x.get_shape().as_list() == [224, 224, 3]
        self.conv1 = conv_2d(x, 96, 11, strides=4, activation='relu')
        self.pool1 = max_pool_2d(self.conv1, 3, strides=2)
        self.conv2 = conv_2d(self.pool1, 256, 5, activation='relu')
        self.pool2 = max_pool_2d(self.conv2, 3, strides=2)
        self.conv3 = conv_2d(self.pool2, 384, 3, activation='relu')
        self.conv4 = conv_2d(self.conv3, 384, 3, activation='relu')
        self.conv5 = conv_2d(self.conv4, 256, 3, activation='relu')
        self.pool5 = max_pool_2d(self.conv5, 3, strides=2)
        self.fc6 = fully_connected(self.pool5, 4096, activation='relu')
        self.fc7 = fully_connected(self.fc6, 4096, activation='relu')
        self.output = fully_connected(self.fc7, 1000, activation='softmax')
        return self.output

model = AlexNetModel()
output = model.build_model(images)
# access intermediate layers like self.conv5, self.fc7, etc.
```


---

## Tensor Fetching: The Bad (ii)

* The feed-forward (sometimes) needs to be done in an atomic way (i.e. a single call of `Session.run()`)
```python
# a single step of training ...
*[loss_value, _] = session.run([loss_op, train_op],
                               feed_dict={images: batch_image})
# After this, the model parameter has been changed due to `train_op`
#
if np.isnan(loss_value):
        # DEBUG : can we see the intermediate values for the current input?
        [fc7, prob] = session.run([net['fc7'], net['prob']],
                                  feed_dict={images: batch_image})
```

* In other words, if any input is fed via `feed_dict`,
  we may have to fetch the non-debugging-related tensors
  and the debugging-related tensors *at the same time*.


---

## Tensor Fetching: The Bad (ii)

- In fact, we can just perform an additional `session.run()` for debugging purposes,
if it does not involve any side effect
```python
# for debugging only, get the intermediate layer outputs.
[fc7, prob] = session.run([net['fc7'], net['prob']],
                               feed_dict={images: batch_image})
#
# Yet another feed-forward: 'fc7' are computed once more ...
[loss_value, _] = session.run([loss_op, train_op],
                               feed_dict={images: batch_image})
```

* A workaround: Use [`session.partial_run()`][tf-partial-run] .gray[(undocumented, and still experimental)]
```python
h = sess.partial_run_setup([net['fc7'], loss_op, train_op], [images])
[loss_value, _] = sess.partial_run(h, [loss_op, train_op],
                                       feed_dict={images: batch_image})
fc7 = sess.partial_run(h, net['fc7'])
```

[tf-partial-run]: https://github.com/tensorflow/tensorflow/blob/v1.0.0/tensorflow/python/client/session.py#L777


---

## (2) Tensorboard

- An off-the-shelf monitoring and debugging tool!
- Check out a must-read [tutorial][tf-tensorboard] from TensorFlow documentation

[tf-tensorboard]: https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html


<br/><br/>

- You will need to learn
  - how to use and collect [scalar/histogram/image summary][apidocs-summary]
  - how to use [`tf.summary.FileWriter`][apidocs-summarywriter] .dogdrip[(previously it was `SummaryWriter`)]

[apidocs-summary]: https://www.tensorflow.org/versions/master/api_docs/python/summary/generation_of_summaries_#scalar
[apidocs-summarywriter]: https://www.tensorflow.org/versions/master/api_docs/python/summary/generation_of_summaries_#FileWriter


---

## Tensorboard: A Quick Tutorial

```python
def multilayer_perceptron(x):
    # inside this, variables 'fc1/weights' and 'fc1/bias' are defined
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu,
                                 scope='fc1')
*   tf.summary.histogram('fc1', fc1)
*   tf.summary.histogram('fc1/sparsity', tf.nn.zero_fraction(fc1))

    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu,
                                 scope='fc2')
*   tf.summary.histogram('fc2', fc2)
*   tf.summary.histogram('fc2/sparsity', tf.nn.zero_fraction(fc2))
    out = layers.fully_connected(fc2, 10, scope='out')
    return out

# ... (omitted) ...
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                      logits=pred, labels=y))
*tf.scalar_summary('loss', loss)

*# histogram summary for all trainable variables (slow?)
*for v in tf.trainable_variables():
*    tf.summary.histogram(v.name, v)
```


---

## Tensorboard: A Quick Tutorial

```python
*global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
train_op = tf.train.AdamOptimizer(learning_rate=0.001)\
                   .minimize(loss, global_step=global_step)
```

```python
def train(session):
    batch_size = 200
    session.run(tf.global_variables_initializer())
*   merged_summary_op = tf.summary.merge_all()
*   summary_writer = tf.summary.FileWriter(FLAGS.train_dir, session.graph)

    # ... (omitted) ...
        for step in range(batch_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
*           _, c, summary = session.run(
                [train_op, loss, merged_summary_op],
                feed_dict={x: batch_x, y: batch_y})
*           summary_writer.add_summary(summary,
*                                      global_step.eval(session=session))
```

---

## Tensorboard: A Quick Tutorial (Demo)

Scalar Summary

.center.img-100[![](images/tensorboard-01-loss.png)]

---
## Tensorboard: A Quick Tutorial (Demo)

Histogram Summary (activations and variables)

.center.img-66[![](images/tensorboard-02-histogram.png)]


---

## Tensorboard and Summary: Noteworthies

- Fetching histogram summary is *extremely* slow!
    - GPU utilization can become very low (if the serialized values are huge)
    - In non-debugging mode, disable it completely; or fetch summaries only **periodically**, e.g.
    ```python
    eval_tensors = [self.loss, self.train_op]
    if step % 200 == 0:
            eval_tensors += [self.merged_summary_op]
        eval_ret = session.run(eval_tensors, feed_dict)
        eval_ret = dict(zip(eval_tensors, eval_ret))  # as a dict

        current_loss = eval_ret[self.loss]
        if self.merged_summary_op in eval_tensors:
            self.summary_writer.add_summary(
                eval_ret[self.merged_summary_op], current_step)
    ```
    - I recommend to take simple and essential scalar summaries *only* (e.g. train/validation loss, overall accuracy, etc.), and to include debugging stuffs only on demand

---

## Tensorboard and Summary: Noteworthies

- Some other recommendations:
    - Use **proper names** (prefixed or scoped) for tensors and variables (specifying `name=...` to tensor/variable declaration)
    - Include both of train loss and validation loss,
      plus train/validation accuracy (if possible) over step

.center.img-50[![](images/tensorboard-02-histogram.png)]

---

## (3) [`tf.Print()`][tf-print]

- During run-time evaluation, we can print the value of a tensor
  .green[without] explicitly fetching and returning it to the code (i.e. via `session.run()`)

[tf-print]: https://www.tensorflow.org/versions/r0.8/api_docs/python/control_flow_ops.html#Print

```python
tf.Print(input, data, message=None,
             first_n=None, summarize=None, name=None)
```

- It creates .blue[an **identity** op] with the side effect of printing `data`,
  when this op is evaluated.

---

## (3) `tf.Print()`: Examples

```python
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu)
    out = layers.fully_connected(fc2, 10, activation_fn=None)
*   out = tf.Print(out, [tf.argmax(out, 1)],
*                  'argmax(out) = ', summarize=20, first_n=7)
    return out
```

For the first seven times (i.e. 7 feed-forwards or SGD steps),
it will print the predicted labels for the 20 out of `batch_size` examples

```x-small
I tensorflow/core/kernels/logging_ops.cc:79] argmax(out) = [6 6 6 4 4 6 4 4 6 6 4 0 6 4 6 4 4 6 0 4...]
I tensorflow/core/kernels/logging_ops.cc:79] argmax(out) = [6 6 0 0 3 6 4 3 6 6 3 4 4 4 4 4 3 4 6 7...]
I tensorflow/core/kernels/logging_ops.cc:79] argmax(out) = [3 4 0 6 6 6 0 7 3 0 6 7 3 6 0 3 4 3 3 6...]
I tensorflow/core/kernels/logging_ops.cc:79] argmax(out) = [6 1 0 0 0 3 3 7 0 8 1 2 0 9 9 0 3 4 6 6...]
I tensorflow/core/kernels/logging_ops.cc:79] argmax(out) = [6 0 0 9 0 4 9 9 0 8 2 7 3 9 1 8 3 9 7 8...]
I tensorflow/core/kernels/logging_ops.cc:79] argmax(out) = [6 0 1 1 9 0 8 3 0 9 9 0 2 6 7 7 3 3 3 9...]
I tensorflow/core/kernels/logging_ops.cc:79] argmax(out) = [3 6 9 8 3 9 1 0 1 1 9 3 2 3 9 9 3 0 6 6...]
[2016-06-03 00:11:08.661563] Epoch 00, Loss = 0.332199
```

---

## (3) `tf.Print()`: Some drawbacks ...

.red[Cons:]

- It is hard to take a full control of print formats (e.g. how do we print a 2D tensor in a matrix format?)
- Usually, we may want to print debugging values **conditionally** <br>(i.e. print them only if some condition is met)
  or **periodically** <br/> (i.e. print just only once per epoch)
    - `tf.Print()` has limitations to achieve this
    - TensorFlow has control flow operations; an overkill?

---

## (3) [`tf.Assert()`][tf-assert]

[tf-assert]: https://www.tensorflow.org/versions/r0.8/api_docs/python/control_flow_ops.html#Assert


* Asserts that the given condition is true, *when evaluated* (during the computation)
* If condition evaluates to `False`, print the list of tensors in `data`,
  and an error is thrown.
  `summarize` determines how many entries of the tensors to print.
```python
tf.Assert(condition, data, summarize=None, name=None)
```

---

## `tf.Assert`: Examples

Abort the program if ...

```python
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, 10, activation_fn=None, scope='out')
    # let's ensure that all the outputs in `out` are positive
*   tf.Assert(tf.reduce_all(out > 0), [out], name='assert_out_positive')
    return out
```


--

The assertion will not work!

- `tf.Assert` is also an op, so it should be executed as well



---

## `tf.Assert`: Examples

We need to ensure that `assert_op` is being executed when evaluating `out`:

```python
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, 10, activation_fn=None, scope='out')
    # let's ensure that all the outputs in `out` are positive
    assert_op = tf.Assert(tf.reduce_all(out > 0), [out], name='assert_out_positive')
*   with tf.control_dependencies([assert_op]):
*       out = tf.identity(out, name='out')
    return out
```

... somewhat ugly? ... or

```python
    # ... same as above ...
*   out = tf.with_dependencies([assert_op], out)
    return out
```


---

## `tf.Assert`: Examples

Another good way: store all the created assertion operations into a collection,
  (merge them into a single op), and explicitly evaluate them using `Session.run()`


```python
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, 10, activation_fn=None, scope='out')
*   tf.add_to_collection('Asserts',
*        tf.Assert(tf.reduce_all(out > 0), [out], name='assert_out_gt_0')
*   )
    return out

# merge all assertion ops from the collection
*assert_op = tf.group(*tf.get_collection('Asserts'))
```

```python
... = session.run([train_op, assert_op], feed_dict={...})
```


---

## Some built-in useful Assert ops

See [Asserts and boolean checks](https://www.tensorflow.org/versions/r1.0/api_docs/python/check_ops.html#asserts-and-boolean-checks) in the docs!

```python
tf.assert_negative(x, data=None, summarize=None, name=None)
tf.assert_positive(x, data=None, summarize=None, name=None)
tf.assert_proper_iterable(values)
tf.assert_non_negative(x, data=None, summarize=None, name=None)
tf.assert_non_positive(x, data=None, summarize=None, name=None)
tf.assert_equal(x, y, data=None, summarize=None, name=None)
tf.assert_integer(x, data=None, summarize=None, name=None)
tf.assert_less(x, y, data=None, summarize=None, name=None)
tf.assert_less_equal(x, y, data=None, summarize=None, name=None)
tf.assert_rank(x, rank, data=None, summarize=None, name=None)
tf.assert_rank_at_least(x, rank, data=None, summarize=None, name=None)
tf.assert_type(tensor, tf_type)
tf.is_non_decreasing(x, name=None)
tf.is_numeric_tensor(tensor)
tf.is_strictly_increasing(x, name=None)
```

If we need runtime assertions during computation, they are useful.

---

## (4) Step-by-step Debugger

Python already has a powerful debugging utilities:

- [`pdb`][pdb]
- [`ipdb`][ipdb]
- [`pudb`][pudb]

which are all **interactive** debuggers (like `gdb` for C/C++)

- set breakpoint
- pause, continue
- inspect stack-trace upon exception
- watch variables and evaluate expressions interactively

[pdb]: https://docs.python.org/2/library/pdb.html
[ipdb]: https://pypi.python.org/pypi/ipdb
[pudb]: https://pypi.python.org/pypi/pudb

---

## Debugger: Usage

Insert `set_trace()` for a breakpoint

```python
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, 10, activation_fn=None, scope='out')
*   import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    return out
```

.gray[(yeah, it is a breakpoint on model building)]

.img-75.center[
![](images/pdb-example-01.png)
]


---

## Debugger: Usage

Debug breakpoints can be conditional:


```python
    for i in range(batch_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
*       if (np.argmax(batch_y, axis=1)[:7] == [4, 9, 6, 2, 9, 6, 5]).all():
*           import pudb; pudb.set_trace()  # XXX BREAKPOINT
        _, c = session.run([train_op, loss],
                            feed_dict={x: batch_x, y: batch_y})
```

.green[Live Demo and Case Example!!] .small[(`20-mnist-pdb.py`)]

- Let's break on training loop if some condition is met
- Get the `fc2` tensor, and fetch its evaluation result (given `x`)
  - `Session.run()` can be invoked and executed anywhere, *even in the debugger*
- .dogdrip[Wow.... gonna love it..]



---

## Hint: Some Useful TensorFlow APIs

To get any .green[operations] or .green[tensors] that might *not* be stored explicitly:

- `tf.get_default_graph()`: Get the current (default) graph
- `G.get_operations()`: List all the TF ops in the graph
- `G.get_operation_by_name(name)`: Retrieve a specific TF op
  <br/> .gray[(Q. How to convert an operation to a tensor?)]
- `G.get_tensor_by_name(name)`: Retrieve a specific tensor
- `tf.get_collection(tf.GraphKeys.~~`): Get the collection of some tensors

To get .green[variables]:

- `tf.get_variable_scope()`: Get the current variable scope
- `tf.get_variable()`: Get a variable (see [Sharing Variables][tensorflow-variable-scope])
- `tf.trainable_variables()`: List all the (trainable) variables
```python
  [v for v in tf.all_variables() if v.name == 'fc2/weights:0'][0]
```

[tensorflow-variable-scope]: https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html

---

## `IPython.embed()`

.green[ipdb/pudb:]

```python
import pudb; pudb.set_trace()
```

- They are debuggers; we can set breakpoint, see stacktraces, watch expressions, ...
  (much more general)

.green[embed:]

```python
from IPython import embed; embed()
```

- Open an ipython shell on the current context;
  mostly used for watching expressions only

<!--
If using [`%pdb` magic][pdb-magic] in IPython notebook:

```python
oops
```

[pdb-magic]: http://ipython.readthedocs.io/en/stable/interactive/magics.html?highlight=magic#magic-pdb
-->


---

## (5) Debugging 'inside' the computation graph

Our debugging tools so far can be used for debugging outside `Session.run()`.

Question: How can we run a .red['custom'] operation? (e.g. custom layer)


--

- TensorFlow allows us [to write a custom operation][docs-custom-ops] in C++ !

- The 'custom' operation can be designed for logging or debugging purposes (like [PrintOp][tf-code-printop])
- ... but very burdensome (need to compile, define op interface, and use it ...)

[docs-custom-ops]: https://www.tensorflow.org/versions/master/how_tos/adding_an_op/index.html#implement-the-kernel-for-the-op
[tf-code-printop]: https://github.com/tensorflow/tensorflow/blob/v1.0.0/tensorflow/core/kernels/logging_ops.cc#L53


---

## (5) Interpose any python code in the computation graph

We can also **embed** and **interpose** a python function in the graph:
[`tf.py_func()`][docs-py-func] comes to the rescue!

```python
tf.py_func(func, inp, Tout, stateful=True, name=None)
```

- Wraps a python function and uses it .blue[**as a tensorflow op**].
- Given a python function `func`, which .green[*takes numpy arrays*] as its inputs and returns numpy arrays as its outputs, the function is wrapped as an operation.

```python
def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    return np.sinh(x)

inp = tf.placeholder(tf.float32, [...])
*y = py_func(my_func, [inp], [tf.float32])
```

[docs-py-func]: https://www.tensorflow.org/versions/r1.0/api_docs/python/script_ops.html#py_func



---

## (5) Interpose any python code in the computation graph

In other words, we are now able to use the following (hacky) .green[**tricks**]
by intercepting the computation being executed on the graph:

- Print any intermediate values (e.g. layer activations) in a custom form
  without fetching them
- Use python debugger (e.g. trace and breakpoint)
- Draw a graph or plot using `matplotlib`, or save images into file

.gray.small[Warning: Some limitations may exist, e.g. thread-safety issue, not allowed to manipulate the state of session, etc..]


---

## Case Example (i): Print

An ugly example ...

```python
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, 10, activation_fn=None, scope='out')

*   def _debug_print_func(fc1_val, fc2_val):
        print 'FC1 : {}, FC2 : {}'.format(fc1_val.shape, fc2_val.shape)
        print 'min, max of FC2 = {}, {}'.format(fc2_val.min(), fc2_val.max())
        return False

*   debug_print_op = tf.py_func(_debug_print_func, [fc1, fc2], [tf.bool])
    with tf.control_dependencies(debug_print_op):
        out = tf.identity(out, name='out')
    return out
```


---

## Case Example (ii): Breakpoints!

An ugly example to attach breakpoints ...

```python
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, 10, activation_fn=None, scope='out')

*   def _debug_func(x_val, fc1_val, fc2_val, out_val):
        if (out_val == 0.0).any():
*           import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
*       from IPython import embed; embed()  # XXX DEBUG
        return False

*   debug_op = tf.py_func(_debug_func, [x, fc1, fc2, out], [tf.bool])
    with tf.control_dependencies(debug_op):
        out = tf.identity(out, name='out')
    return out
```


---

## Another one: The `tdb` library

A third-party TensorFlow debugging tool: .small[https://github.com/ericjang/tdb]
<br/> .small[(not actively maintained and looks clunky, but still good for prototyping)]

.img-90.center[
![](https://camo.githubusercontent.com/4c671d2b359c9984472f37a73136971fd60e76e4/687474703a2f2f692e696d6775722e636f6d2f6e30506d58516e2e676966)
]



---

## (6) `tfdbg`: The *official* TensorFlow debugger

<!--.small.green.emph[(Added in December 2016)]-->

Recent versions of TensorFlow has the official debugger ([`tfdbg`](https://www.tensorflow.org/programmers_guide/debugger)).
Still experimental, but works quite well!

Check out the [HOW-TOs](https://www.tensorflow.org/programmers_guide/debugger) and [Examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/debug/examples) on `tfdbg`!!!

```python
import tensorflow.python.debug as tf_debug
sess = tf.Session()

# create a debug wrapper session
*sess = tf_debug.LocalCLIDebugWrapperSession(sess)

# Add a tensor filter (similar to breakpoint)
*sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

# Each session.run() will be intercepted by the debugger,
# and we can inspect the value of tensors via the debugger interface
sess.run(loss, feed_dict = {x : ...})
```

<!--
Although it is not yet fully functional and has some bugs, it is quite usable!
(Google Brain team will complete and announce it soon)
-->


---

## (6) `tfdbg`: The TensorFlow debugger

<!--.small.green.emph[(Added in December 2016)]-->

.img-100.center[
![](images/tfdbg_example1.png)
]

---

## (6) `tfdbg`: The TensorFlow debugger

<!--.small.green.emph[(Added in December 2016)]-->

.img-100.center[
![](images/tfdbg_example2.png)
]

<!--
## Teaser: `tfdb`

A new TensorFlow debugger and helper library will be published soon :-)

```python
import tfdb

def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, 10, activation_fn=None, scope='out')

*   out = tfdb.with_breakpoint([x, fc1, fc2, out], out)  # XXX BREAKPOINT
    return out
```

-->


---

## `tfdbg`: Features and Quick References

Conceptually, a wrapper session is employed (currently, CLI debugger session); it can intercept a single run of `session.run()`


- `run` / `r` : Execute the run() call .blue[with debug tensor-watching]
- `run -n` / `r -n` : Execute the run() call .blue[without] debug tensor-watching
- `run -f <filter_name>` : Keep executing run() calls until a dumped tensor passes a registered filter (conditional breakpoint)
    - e.g. `has_inf_or_nan`

.img-80.center[
![](images/tfdbg_example_run.png)
]

---

## `tfdbg`: Tensor Filters

Registering tensor filters:

```python
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
```

Tensor filters are just python functions `(datum, tensor) -> bool`:

```python
def has_inf_or_nan(datum, tensor):
  _ = datum  # Datum metadata is unused in this predicate.

  if tensor is None:
    # Uninitialized tensor doesn't have bad numerical values.
    return False
  elif (np.issubdtype(tensor.dtype, np.float) or
        np.issubdtype(tensor.dtype, np.complex) or
        np.issubdtype(tensor.dtype, np.integer)):
    return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
  else:
    return False
```
Running tensor filters are, therefore, quite slow.


---

## `tfdbg`: Tensor Fetching

In a tensor dump mode (the **run-end UI**), the debugger shows the list of tensors dumped in the `session.run()` call:

.img-100.center[
![](images/tfdbg_example_fetch.png)
]

---

## `tfdbg`: Tensor Fetching

Commands:

- .blue[`list_tensors` (`lt`)] : Show the list of dumped tensor(s).
- .blue[`print_tensor` (`pt`)] : Print the value of a dumped tensor.
- `node_info` (`ni`) : Show information about a node
    - `ni -t` : Shows the traceback of tensor creation
- `list_inputs` (`li`) : Show inputs to a node
- `list_outputs` (`lo`) : Show outputs to a node
- `run_info` (`ri`) : Show the information of current run <br/> (e.g. what to fetch, what feed_dict is)
- .green[`invoke_stepper` (`s`)] : Invoke the stepper!
- `run` (`r`) : Move to the next run


---

## `tfdbg`: Tensor Fetching

Example: `print_tensor fc2/Relu:0`

.img-80.center[
![](images/tfdbg_example_pt.png)
]

- Slicing: `pt f2/Relu:0[0:10]`
- Dumping: `pt fc2/Relu:0 > /tmp/debug/fc2.txt`

**See also**: [tfdbg CLI Frequently-Used Commands](https://www.tensorflow.org/programmers_guide/debugger#tfdbg_cli_frequently-used_commands)



---

## `tfdbg`: Stepper

Shows the tensor value(s) in a topologically-sorted order for the run.

.img-100.center[
![](images/tfdbg_example_stepper.png)
]



---

## `tfdbg`: Screencast and Demo!

.small.right[From Google Brain Team]

<div class="center">
<iframe width="672" height="378" src="https://www.youtube.com/embed/CA7fjRfduOI" frameborder="0" allowfullscreen></iframe>
</div>

<p>

.small[
<br/>
See also: [Debug TensorFlow Models with tfdbg (@Google Developers Blog)](https://developers.googleblog.com/2017/02/debug-tensorflow-models-with-tfdbg.html)
]

---

## `tfdbg`: Other Remarks

- Currently it is actively being developed (still experimental)
- In a near future, a web-based interactive debugger (integration with TensorBoard) will be out!

---

## Debugging: Summary

* `Session.run()`: Explicitly fetch, and print
* Tensorboard: Histogram and Image Summary
* `tf.Print()`, `tf.Assert()` operation
* Use python debugger (`ipdb`, `pudb`)
* Interpose your debugging python code in the graph
* The TensorFlow debugger: `tfdbg`


.green[There is no silver bullet; one might need to choose the most convenient and suitable debugging tool, depending on the case]

---

template: inverse

# Other General Tips

## .gray[(in a Programmer's Perspective)]

---

## General Tips of Debugging

- Learn to use debugging tools, but do not solely rely on them when a problem occurs.
- Sometimes, just sitting down and reading through 👀 your code with ☕ (a careful code review!) would be greatly helpful.

---

## General Tips from Software Engineering

Almost .red[all] of rule-of-thumb tips and guidelines for writing good, neat, and defensive codes can be applied to TensorFlow codes :)

* Check and sanitize inputs
* Logging
* Assertions
* Proper usage of Exception
* [Fail Fast][fail-fast]: immediately abort if something is wrong
* [The DRY principle][dry-principle]: Don't Repeat Yourself
* Build up well-organized codes, test smaller modules
* etc ...

[fail-fast]: https://en.wikipedia.org/wiki/Fail-fast
[dry-principle]: https://en.wikipedia.org/wiki/Don%27t_repeat_yourself

<p>

There are some good guides on the web like [this][debug-tip-matloff]

[debug-tip-matloff]: http://heather.cs.ucdavis.edu/~matloff/UnixAndC/CLanguage/Debug.html

---

## Use asserts as much as you can

* Use assertion anywhere (early fail is always good)
  * e.g. Data processing routine (input sanity check)
* Especially, perform .red[shape check] for tensors (like 'static' type checking when compiling codes)

```python
net['fc7'] = tf.nn.xw_plus_b(net['fc6'], vars['fc7/W'], vars['fc7/b'])

assert net['fc7'].get_shape().as_list() == [None, 4096]
*net['fc7'].get_shape().assert_is_compatible_with([B, 4096])
```

* Sometimes, `tf.Assert()` operation might be helpful (a run-time checking)
  * should be turned off if we are not debugging now

---

## Use proper logging

- Being verbose for logging helps a lot (configurations for training hyperparameters, monitor train/validation loss, learning rate, elapsed time, etc.) <br/><br/>
.img-100.center[
![](images/logging-example.png)
]

---

## Guard against Numerical Errors

Quite often, `NaN` occurs during the training
  * We usually deal with `float32` which is not a precise datatype;
    deep learning models are susceptible to numerical instability
  * Some possible reasons:
    * gradient is too big <span class="gray">(clipping may be required)</span>
    * zero or negatives are passed in `sqrt` or `log`
  * Check: whether it is `NaN` or is finite

<p>

* Some useful TF APIs
  * .small[`loss = [tf.verify_tensor_all_finite(loss, msg)`](https://www.tensorflow.org/versions/master/api_docs/python/control_flow_ops.html#verify_tensor_all_finite)]
  * .small[[`tf.add_check_numerics_ops()`](https://www.tensorflow.org/versions/master/api_docs/python/control_flow_ops.html#add_check_numerics_ops)]


---

## Name your tensors properly

* It is recommended to **specify names** for intermediate tensors and variables when building model
    * Using variable scopes properly is also a very good idea
* When something wrong happens, we can easily figure out where the error is from

```python
ValueError: Cannot feed value of shape (200,)
      for Tensor u'Placeholder_1:0', which has shape '(?, 10)'
ValueError: Tensor conversion requested dtype float32 for Tensor with
*     dtype int32: 'Tensor("Variable_1/read:0", shape=(256,), dtype=int32)'
```

A better stacktrace:

```python
ValueError: Cannot feed value of shape (200,)
      for Tensor u'placeholder_y:0', which has shape '(?, 10)'
ValueError: Tensor conversion requested dtype float32 for Tensor with
*     dtype int32: 'Tensor("fc1/weights/read:0", shape=(256,), dtype=int32)'
```

---

## Name your tensors properly

```python
def multilayer_perceptron(x):
    W_fc1 = tf.Variable(tf.random_normal([784, 256], 0, 1))
    b_fc1 = tf.Variable([0] * 256) # wrong here!!
    fc1 = tf.nn.xw_plus_b(x, W_fc1, b_fc1)
    # ...
```

```python
>>> fc1
<tf.Tensor 'xw_plus_b:0' shape=(?, 256) dtype=float32>
```

Better:
```python
def multilayer_perceptron(x):
    W_fc1 = tf.Variable(tf.random_normal([784, 256], 0, 1), name='fc1/weights')
    b_fc1 = tf.Variable(tf.zeros([256]), name='fc1/bias')
    fc1 = tf.nn.xw_plus_b(x, W_fc1, b_fc1, name='fc1/linear')
    fc1 = tf.nn.relu(fc1, name='fc1/relu')
    # ...
```

```python
>>> fc1
<tf.Tensor 'fc1/relu:0' shape=(?, 256) dtype=float32>
```

---

## Name your tensors properly

The style that I much prefer:

```python
def multilayer_perceptron(x):
*   with tf.variable_scope('fc1'):
        W_fc1 = tf.get_variable('weights', [784, 256])  # fc1/weights
        b_fc1 = tf.get_variable('bias', [256])          # fc1/bias
        fc1 = tf.nn.xw_plus_b(x, W_fc1, b_fc1)          # fc1/xw_plus_b
        fc1 = tf.nn.relu(fc1)                           # fc1/relu
    # ...
```

or use high-level APIs or your custom functions:

```python
import tensorflow.contrib.layers as layers

def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu,
*                                scope='fc1')
    # ...
```

---

## And more style guides ...?

<div class="center border-dotted" style="padding: 2em 0; margin: 3em 0;">
.large[Toward Best Practices of TensorFlow Code Patterns]

.small[https://github.com/wookayin/TensorFlowKR-2017-talk-bestpractice]
</div>

all of which will help you to write easily-debuggable codes!

---

## Other Topics: Performance and Profiling

- Run-time performance is a very important topic! <br/>
  .dogdrip[there will be another lecture soon...]
    - Beyond the scope of this talk...


- Make sure that your GPU utilization is *always* non-zero (and, near 100%)
    - Watch and monitor using `nvidia-smi` or [`gpustat`][gpustat]

.img-75.center[
![](https://github.com/wookayin/gpustat/raw/master/screenshot.png)
]
.img-50.center[
![](images/nvidia-smi.png)
]

[gpustat]: https://github.com/wookayin/gpustat

---

## Other Topics: Performance and Profiling

- Some possible factors that might slow down your code:
    - Input batch preparation (i.e. `next_batch()`)
    - Too frequent or heavy summaries
    - Inefficient model (e.g. CPU-bottlenecked operations)

<p>

- What we can do
    - Use `tfdbg` !!!
    - Use [`cProfile`][python-profilers],
      [`line_profiler`][line-profiler]
      or [`%profile`][ipython-profile] in IPython
    - Use [`nvprof`][nvprof] for profiling CUDA operations
    - Use CUPTI (CUDA Profiling Tools Interface) [tools][tf-issue-1824] for TF

.img-66.center[![](images/tracing-cupti.png)]

[python-profilers]: https://docs.python.org/2/library/profile.html
[line-profiler]: https://pypi.python.org/pypi/line_profiler/
[ipython-profile]: https://ipython.org/ipython-doc/3/interactive/magics.html
[nvprof]: http://docs.nvidia.com/cuda/profiler-users-guide/
[tf-issue-1824]: https://github.com/tensorflow/tensorflow/issues/1824


---

## Concluding Remarks

We have talked about

- How to debug TensorFlow and python applications
- Some tips and guidelines for easy debugging --- write a nice code that would require less debugging :)



---

name: last-page
class: center, middle, no-number

<div style="position:absolute; left:0; bottom:0; padding: 25px;">
  <p class="left" style="margin:0; font-size: 13pt;">
  <b>Special Thanks to</b>: <br/>
  Juyong Kim, Yunseok Jang, Junhyug Noh, Cesc Park, Jongho Park</p>

  <p class="left" style="margin: 10px 0 0; font-size: 11pt; color: gray">
  TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.
  </p>
</div>

.img-66[![](images/tensorflow-logo.png)]

---
name: last-page
class: center, middle, no-number

## Thank You!
#### [@wookayin][wookayin-gh]

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]

<!-- vim: set ft=pandoc -->
