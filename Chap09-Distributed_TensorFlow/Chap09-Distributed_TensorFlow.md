
# Chap09 - 분산 텐서플로

 

## 9.1 분산 컴퓨팅

텐서플로에서의 분산 컴퓨팅은 딥러닝 모델의 학습 속도를 향상 시키기 위해서 여러 대의 기계를 사용하는 것을 말한다. 

 

## 9.2 텐서플로의 병렬처리 요소

그렇다면, 텐서플로에서 병렬계산(parallel computation)에 사용되는 요소와 개념에 대해 알아보도록 하자.

 

### 9.2.1 tf.app.flags

`tf.app.flags`는 병렬계산과는 전혀 상관이 없지만, 텐서플로 예제에서 많이 사용되므로 아는것이 좋다. 그리고 이 교재의 마지막 예제에서도 `tf.app.flags`를 사용한다.

`tf.app.flags`는 Python의 `argparse` 모듈의 래퍼(wrapper)이다. `argparse`모듈은 Terminal이사 cmd창에서의 `python ~.py --~~`에 있는 인자를 처리하는데 사용된다.

예를 들어 아래와 같은 Python 명령을 입력한다고 가정한다면, `distribute.py` 프로그램에 전달되는 인자는 다음과 같다.

```bash
python distribute_mnist_cnn.py --job_name="ps" --task_index=0

##전달되는 인자
job_name="ps"
task_index=0
```

이러한 명령을 통해 다음과 같이 파이썬 스크립트 안에서 이 정보를 추출할 수 있다.

```python
tf.app.flags.DEFINE_string("job_name", "", "name of job")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")
```

`tf.app.flags`에는 다음과 같은 유형들이 있다.

- `tf.app.flags.DEFINE_string`: 스트링 값을 정의
- `tf.app.flags.DEFINE_boolean`: 참거짓 값을 정의
- `tf.app.flags.DEFINE_float`: 부동소수점 값을 정의
- `tf.app.flags.DEFINE_integer`: 정수를 정의


`tf.app.flags.FLAGS`는 명령에서 입력되어 파싱된 모든 인수의 값을 포함하며, `FLAGS.arg`로 접근할 수 있다.

 

### 9.2.2 클러스터와 서버

텐서플로의 클러스터(cluster)는 연산 그래프의 병렬처리에 참여하는 노드(또는 태스크)의 집합이다. 각 태스크는 다음과 같이 접근할 수 있는 네트워크 주소로 정의한다.

```python
parameter_servers = ['hostname1:port']  # ex. localhost:2222
workers = ['hostname2:port',  # ex. 192.168.0.3:2223
           'hostname3:port',
           'hostname4:port']
cluster = tf.train.ClusterSpec({
    'parameter_server': parameter_servers,
    'worker': workers
})
```

위의 예제코드에서 태스크는 하나의 **매개변수 서버(parameter server)**와 세 개의 **워커(worker)**로 구성되어 있으며, 매개변수 서버와 워커의 역할은 **잡(job)**이라고 부른다.

각 태스크는 텐서플로 서버를 실행하며, 계산을 위해 자원을 사용하고 병렬처리를 효율적으로 하도록 클러스터 내의 다른 태스크와 통신한다. 

`hostname2:port`인 워커 노드에 클러스터를 정의하는 방법은 다음과 같다.

```python
server = tf.train.Server(cluster,
                         job_name='worker',
                         task_index=0)
```

위에서 처럼 클러스터와 서버를 정의를 해준 다음, 병렬 계산을 수행할 수 있는 연산 그래프를 작성해야 한다.

 

### 9.2.3 디바이스 간 연산 그래프 복제

**그래프 간 복제(between-graph replication)** 는 동일한 연산 그래프가 각 워크 태스크상에 개별적으로 구성되어있는 병렬화 방법이다. 학슥 단계에서의 기울기(gradient) 값은 각 워커(worker)에 계산되고 매개변수 서버에 의해 결합된다. 

`tf.train.replica_device_setter()`를 이용해 각 태스크에 연산 그래프(모델)을 복제할 수 있다. `worker_device` 인자에 클러스터의 태스크를 설정해준다.

```python
with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % 0,
        cluster=cluster)
    # build model...
```

매개변수 서버에는 연산 그래프(모델)를 작성하지 않으며, 대신 프로세스가 종료되지 않도록 `server.join()`으로 병렬 계산이 수행되는 동안 매개변수 서버가 종료되지 않도록 해준다.

 

### 9.2.4 관리 세션

`tf.train.Supervisor()`는 학습을 관리해주고 병렬 환경을 설정하는데 필요한 기능을 제공한다.

```python
sv = tf.train.Supervisor(is_chief=True,
                         logdir=None,
                         global_step=...,
                         init_op=...)
```

- `is_chief(bool)`: If True, create a chief supervisor in charge of initializing and restoring the model.
- `logdir(string)`: A string. Optional path to a directory where to checkpoint the model and log events for the visualizer. 로그를 저장할 경로.
- `global_step`: An integer Tensor of size 1 that counts steps. 전역 step의 값
- `init_op`: Used by chief supervisors to initialize the model when it can not be recovered. ex. `tf.global_variables_initializer()`

세션은 `Supervisor`의 `managed_session`으로 설정한다.

```python
with sv.managed_session(server.target) as sess:
    # Training
```

 

### 9.2.5 디바이스 배치

텐서플로의 **디바이스 배치(device placement)** 는 CPU 또는 GPU(가 있는경우)에 연산 그래프의 연산이 실행되는 위치를 지정해주는 것을 말한다. 기본적으로 텐서플로는 사용 가능한 모든 CPU를 사용한다. 

텐서플로는 `tf.ConfigProto(log_device_placement=True)`를 통해 연산 그래프가 어떤 디바이스에 배치되었는지 확인할 수 있다. 아래의 예제코드는 연산 그래프가 어떤 디바이스에 배치되었는지 확인하는 코드이다.


```python
# tf_device_example.py
import tensorflow as tf

# generate Graph
a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[3., 4.], [1., 2.]])
c = tf.matmul(a, b)

with tf.Session(
    config=tf.ConfigProto(log_device_placement=True)) as sess:
    res = sess.run(c)
```

```
2018-07-04 15:41:48.549266: I T:\src\github\tensorflow\tensorflow\core\common_runtime\direct_session.cc:284] Device mapping:

MatMul: (MatMul): /job:localhost/replica:0/task:0/device:CPU:0
2018-07-04 15:41:48.553920: I T:\src\github\tensorflow\tensorflow\core\common_runtime\placer.cc:886] MatMul: (MatMul)/job:localhost/replica:0/task:0/device:CPU:0
Const_1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2018-07-04 15:41:48.559578: I T:\src\github\tensorflow\tensorflow\core\common_runtime\placer.cc:886] Const_1: (Const)/job:localhost/replica:0/task:0/device:CPU:0
Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2018-07-04 15:41:48.567907: I T:\src\github\tensorflow\tensorflow\core\common_runtime\placer.cc:886] Const: (Const)/job:localhost/replica:0/task:0/device:CPU:0
[[ 5.  8.]
 [13. 20.]]
```

 

디바이스를 명시적으로 지정해주는 방법은 아래(첫 번째 GPU에 지정)와 같다.

```python
with tf.device('/gpu:0'):
    op = ...
```

클러스터에서의 배치는 특정 태스크를 지정해줘야 한다.

```python
with tf.device('/job:worker/task:0'):  # 첫번째 워커 태스크에 배치
    op = ...
```

 

## 9.3 MNIST CNN 모델을 분산 학습 시키기


이제 앞에서 살펴본 분산 컴퓨팅 방법을 적용해 MNIST 데이터를 분류하는 CNN 모델을 분산 학습을 시켜보도록 하자. 

전체 코드는 아래와 같다.

```python
# ditribute_mnist_cnn.py

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
```

 

위의 코드를 실행하기 위해서는 아래의 `distribute_mnist_cnn-run.py` 파일을 새로운 Terminal에서 실행하면 된다. 

```python
# distribute_mnist_cnn-run.py

import subprocess
subprocess.Popen('python distribute_mnist_cnn.py --job_name="ps" --task_index=0',
                 shell=True)
subprocess.Popen('python distribute_mnist_cnn.py --job_name="worker" --task_index=0',
                 shell=True)
subprocess.Popen('python distribute_mnist_cnn.py --job_name="worker" --task_index=1',
                 shell=True)
subprocess.Popen('python distribute_mnist_cnn.py --job_name="worker" --task_index=2',
                 shell=True)
```

```
Worker : 1, Step: 4400.0, Accuracy (batch): 0.9800000190734863
Worker : 2, Step: 4500.0, Accuracy (batch): 0.9800000190734863
Worker : 0, Step: 4500.0, Accuracy (batch): 1.0
Worker : 2, Step: 4600.0, Accuracy (batch): 0.9800000190734863
Worker : 0, Step: 4600.0, Accuracy (batch): 1.0
Worker : 0, Step: 4800.0, Accuracy (batch): 1.0
Worker : 2, Step: 4900.0, Accuracy (batch): 1.0
Test-Accuracy: 0.9871000051498413
```
