import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

############################
# 0. mnist 불러오기
def mnist_load():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    # Train - Image
    train_x = train_x.astype('float32') / 255
    train_x = train_x.reshape(-1, 28*28)
    # Train - Label(OneHot)
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)

    # Test - Image
    test_x = test_x.astype('float32') / 255
    test_x = test_x.reshape(-1, 28*28)
    # Test - Label(OneHot)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=10)

    return (train_x, train_y), (test_x, test_y)

(train_x, train_y), (test_x, test_y) = mnist_load()


#######################################
# 1. Hyper parameter 및 Cluster 정의

# Hyper-Parameter
BATCH_SIZE = 50  # mini-batch
TRAINING_STEPS = 5000
PRINT_EVERY = 100
LOG_DIR = "./logs"

# 예제이므로 분산 작업은 로컬에서 실행된다.
# port번호 2222~2225는 임의로 설정해준 것이며,
# hostname이 localhost로 같기 때문에 port 번호를 다르게 설정해야함
parameter_servers = ["localhost:2222"]
workers = ["localhost:2223",
           "localhost:2224",
           "localhost:2225"]

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")
FLAGS = tf.app.flags.FLAGS


server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)


#########################################
# 2. TF-Slim을 이용한 CNN 모델링
def net(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    net = slim.layers.conv2d(x_image, 32, [5, 5], scope='conv1')
    net = slim.layers.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.layers.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.layers.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.layers.flatten(net, scope='flatten')
    net = slim.layers.fully_connected(net, 500, scope='fully_connected')
    net = slim.layers.fully_connected(net, 10, activation_fn=None, scope='pred')
    return net


########################################
# 3. 분산 학습

# Mini-Batch Dataset
dataset = tf.data.Dataset.from_tensor_slices(({"image": train_x}, train_y))
dataset = dataset.shuffle(100000).repeat().batch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

if FLAGS.job_name == "ps":
    server.join()

elif FLAGS.job_name == "worker":

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
        y = net(x)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,
                                                                               labels=y_))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,
                                                           global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir=LOG_DIR,
                             global_step=global_step,
                             init_op=init_op)

    with sv.managed_session(server.target) as sess:
        step = 0

        while not sv.should_stop() and step <= TRAINING_STEPS:

            batch_x, batch_y = sess.run(next_batch)

            _, acc, step = sess.run([train_step, accuracy, global_step],
                                    feed_dict={x: batch_x['image'], y_: batch_y})

            if step % PRINT_EVERY == 0:
                print("Worker : {}, Step: {}, Accuracy (batch): {}".
                      format(FLAGS.task_index, step, acc))

        test_acc = sess.run(accuracy,
                            feed_dict={x: test_x, y_: test_y})
        print("Test-Accuracy: {}".format(test_acc))

    sv.stop()