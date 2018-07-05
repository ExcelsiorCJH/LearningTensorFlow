
# Chap10.1 - 모델 익스포트와 서빙, Saver

> 학습한 모델을 저장하고 내보내는 방법에 대해 NumPy의 `.savez()`와 텐서플로의 `Saver`를 사용해 학습된 가중치를 저장하고 로드해보자.

 

## 10.1 모델을 저장하고 내보내기

텐서플로를 이용해 모델을 만들고 학습한 뒤 학습된 모델 즉, 매개변수(weight, bias)를 저장하는 방법에 대해 알아보자. 이렇게 학습된 모델을 저장해놓으면 나중에 모델을 처음 부터 다시 학습시킬 필요가 없기 때문에 편리하다.

학습된 모델을 저장하기 위해 NumPy를 이용해 매개변수를 저장하는 방법을 알아보고, 텐서플로의 `Saver`를 이용해 모델을 저장하고 관리하는 방법에 대해 알아보자. 

 

### 10.1.1 로딩된 가중치 할당 

먼저, NumPy의 `savez`를 이용해 학습된 가중치 값을 저장하고, 불러오는 방법에 대해 알아보자. 이를 위해, [Chap02-텐서플로 설치 및 실행](http://excelsior-cjh.tistory.com/149?category=940399)에서 살펴본 Softmax Regression을 이용해 MNIST 데이터 분류 모델을 만들어 준다.


```python
import tensorflow as tf

# MNIST data load
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x, test_x = train_x.reshape(-1, 28*28), test_x.reshape(-1, 28*28)
train_x, test_x = train_x/255., test_x/255.
train_y = tf.keras.utils.to_categorical(train_y, 10)
test_y = tf.keras.utils.to_categorical(test_y, 10)

inputs = tf.placeholder(tf.float32, [None, 28*28])
weights = tf.Variable(tf.truncated_normal(shape=[28*28, 10], stddev=0.01))

logits = tf.matmul(inputs, weights)
labels = tf.placeholder(tf.float32, [None, 10])

# loss
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

# optimizer
train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# accuracy
correct_mask = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# Hyper-Parameter
NUM_STEPS = 1000
MINIBATCH_SIZE = 100
TRAIN_SIZE = train_x.shape[0]

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1, NUM_STEPS+1):
        # mini-batch using numpy
        batch_mask = np.random.choice(TRAIN_SIZE, MINIBATCH_SIZE)
        batch_x, batch_y = train_x[batch_mask], train_y[batch_mask]
        loss, _ = sess.run([cross_entropy, train_op], feed_dict={inputs:batch_x, 
                                                                 labels:batch_y})
        
        if step % 200 == 0:
            test_acc = sess.run(accuracy, feed_dict={inputs: test_x, 
                                                     labels: test_y})
            print('Step: {:04d}, Loss: {:.5f}, Test_Acc: {:.5f}'.format(step, loss, test_acc))
            
    # 학습된 가중치를 res_weights에 할당하기
    res_weights = sess.run(weights)
print("최적화 완료")

# 출력결과
# Step: 0200, Loss: 0.59659, Test_Acc: 0.87730
# Step: 0400, Loss: 0.47516, Test_Acc: 0.89680
# Step: 0600, Loss: 0.37015, Test_Acc: 0.90610
# Step: 0800, Loss: 0.37617, Test_Acc: 0.91050
# Step: 1000, Loss: 0.32156, Test_Acc: 0.91150
# 최적화 완료
```




학습이 끝났으므로, 학습된 가중치를 NumPy의 `savez()`를 이용해 저장해주자. `savez()`는 NumPy의 array형식을 `.npz`파일로 저장해주는 기능을 한다.


```python
import os
import numpy as np

MODEL_PATH = './model/'
np.savez(os.path.join(MODEL_PATH, 'weight_storage'), res_weights)
```

 

#### 저장된 가중치 로드하기

위의 코드를 통해 저장된 `weight_storage.npz`파일을 불러와 `tf.Variable()`의 `.assign()`메소드를 통해 학습된 가중치들을 할당해줄 수 있다.

아래의 코드는 위에서 구현한 Softmax Regression을 학습된 가중치를 가지고 정확도(`accuracy`)를 구하는 코드이다.


```python
import numpy as np
import tensorflow as tf

# 학습된 가중치 불러오기
MODEL_PATH = './model/'
loaded_w = np.load(MODEL_PATH + 'weight_storage.npz')
loaded_w = loaded_w.items()[0][1]

# MNIST data load
_, (test_x, test_y) = tf.keras.datasets.mnist.load_data()
test_x = test_x.reshape(-1, 28*28) / 255.
test_y = tf.keras.utils.to_categorical(test_y, 10)

# placeholder and variable
inputs = tf.placeholder(tf.float32, [None, 28*28])
weights = tf.Variable(tf.truncated_normal(shape=[28*28, 10], stddev=0.01))

logits = tf.matmul(inputs, weights)
labels = tf.placeholder(tf.float32, [None, 10])

# accuracy
correct_mask = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# Test
with tf.Session() as sess:
    # 로드한 가중치 할당
    sess.run(weights.assign(loaded_w))
    acc = sess.run(accuracy, feed_dict={inputs: test_x,
                                        labels: test_y})
print("Accuarcy: {:.5f}".format(acc))

# 출력결과
# Accuarcy: 0.91150
```






#### CNN 모델 학습된 가중치 저장 및 로드하기

이번에는 간단한 CNN 모델을 가지고 위와 동일한 방법으로 NumPy의 `cnn_weight_storage.npz`파일로 가중치를 저장한 뒤 로드해 테스트를 해보자.

먼저, CNN 모델을 클래스 형태로 구성해준다.


```python
import tensorflow as tf

# CNN 모델 구현
class SimpleCNN:
    def __init__(self, inputs, 
                 keep_prob, weights=None, sess=None):
        
        self.parameters = []
        self.inputs = inputs
        
        conv1 = self.conv_layer(inputs, shape=[5, 5, 1, 32])
        conv1_pool = self.max_pool_2x2(conv1)
        
        conv2 = self.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
        conv2_pool = self.max_pool_2x2(conv2)
        
        conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
        full_1 = tf.nn.relu(self.full_layer(conv2_flat, 1024))
        
        full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
        
        self.y_conv = self.full_layer(full1_drop, 10)
        
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
            
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name='weights')
    
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name='bias')
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                            padding='SAME')
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                              strides=[1, 2, 2, 1], padding='SAME')
    
    def conv_layer(self, input_, shape):
        W = self.weight_variable(shape)
        b = self.bias_variable([shape[3]])
        self.parameters += [W, b]
        
        return tf.nn.relu(self.conv2d(input_, W) + b)
    
    def full_layer(self, input_, size):
        in_size = int(input_.get_shape()[1])
        W = self.weight_variable([in_size, size])
        b = self.bias_variable([size])
        self.parameters += [W, b]
        return tf.matmul(input_, W) + b
    
    def load_weights(self, weights, sess):
        for idx, w in enumerate(weights):
            print("Weight index: {}".format(idx),
                  "Weigth shape: {}".format(w.shape))
            sess.run(self.parameters[idx].assign(w))
```

 

구현한 `SimpleCNN` 클래스를 이용해 학습을 하고 가중치를 저장한다.


```python
import numpy as np
import tensorflow as tf

# MNIST data load
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x, test_x = train_x.reshape(-1, 28*28), test_x.reshape(-1, 28*28)
train_x, test_x = train_x/255., test_x/255.
train_y = tf.keras.utils.to_categorical(train_y, 10)
test_y = tf.keras.utils.to_categorical(test_y, 10)

# placeholder
inputs = tf.placeholder(tf.float32, shape=[None, 28*28])
x_image = tf.reshape(inputs, [-1, 28, 28, 1])
labels = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# Hyper-Parameter
NUM_STEPS = 1000
MINIBATCH_SIZE = 100
TRAIN_SIZE = train_x.shape[0]

with tf.Session() as sess:
    # CNN model
    cnn = SimpleCNN(x_image, keep_prob, sess)
    
    # loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=cnn.y_conv, 
                                                   labels=labels))
    # optimizer
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    # accuracy
    correct_prediction = tf.equal(tf.argmax(cnn.y_conv, 1), 
                                  tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    # Train
    for step in range(1, NUM_STEPS+1):
        # mini-batch using numpy
        batch_mask = np.random.choice(TRAIN_SIZE, MINIBATCH_SIZE)
        batch_x, batch_y = train_x[batch_mask], train_y[batch_mask]
        
        loss, _ = sess.run([cross_entropy, train_op], feed_dict={inputs: batch_x, 
                                                                 labels: batch_y, 
                                                                 keep_prob: 1.0})
        if step % 200 == 0:
            test_acc = sess.run(accuracy, feed_dict={inputs: test_x, 
                                                     labels: test_y,
                                                     keep_prob: 1.0})
            print('Step: {:04d}, Loss: {:.5f}, Test_Acc: {:.5f}'.format(step, loss, test_acc))
            
    # 학습된 가중치를 res_weights에 할당하기
    res_weights = sess.run(cnn.parameters)
print("최적화 완료")

# Save Weights using np.savez
MODEL_PATH = './model/'
np.savez(os.path.join(MODEL_PATH, 'cnn_weight_storage'), res_weights)

# 출력결과
# Step: 0200, Loss: 0.17530, Test_Acc: 0.91740
# Step: 0400, Loss: 0.24124, Test_Acc: 0.94830
# Step: 0600, Loss: 0.09976, Test_Acc: 0.95210
# Step: 0800, Loss: 0.05825, Test_Acc: 0.96590
# Step: 1000, Loss: 0.10787, Test_Acc: 0.97050
# 최적화 완료
```






이번에는 저장한 `cnn_weight_storage.npz`를 로드하여 학습된 가중치를 이용해 테스트셋을 분류해보자.


```python
import numpy as np
import tensorflow as tf

# MNIST data load
_, (test_x, test_y) = tf.keras.datasets.mnist.load_data()
test_x = test_x.reshape(-1, 28*28) / 255.
test_y = tf.keras.utils.to_categorical(test_y, 10)

# placeholder
inputs = tf.placeholder(tf.float32, shape=[None, 28*28])
x_image = tf.reshape(inputs, [-1, 28, 28, 1])
labels = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# 학습된 가중치 불러오기
MODEL_PATH = './model/'
weights = np.load(MODEL_PATH + 'cnn_weight_storage.npz')
weights = weights.items()[0][1]

with tf.Session() as sess:
    cnn = SimpleCNN(x_image, keep_prob, weights, sess)
    
    # accuracy
    correct_prediction = tf.equal(tf.argmax(cnn.y_conv, 1), 
                                  tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    test_acc = sess.run(accuracy, feed_dict={inputs: test_x, 
                                             labels: test_y,
                                             keep_prob: 1.0})
    print("Test Accuracy: {:.5f}".format(test_acc))
    
# 출력결과
# Weight index: 0 Weigth shape: (5, 5, 1, 32)
# Weight index: 1 Weigth shape: (32,)
# Weight index: 2 Weigth shape: (5, 5, 32, 64)
# Weight index: 3 Weigth shape: (64,)
# Weight index: 4 Weigth shape: (3136, 1024)
# Weight index: 5 Weigth shape: (1024,)
# Weight index: 6 Weigth shape: (1024, 10)
# Weight index: 7 Weigth shape: (10,)
# Test Accuracy: 0.97050
```






### 10.1.2 Saver 클래스

텐서플로는 자체적으로 학습된 모델을 저장하고 로드할 수 있는 기능인 `Saver`라는 클래스를 제공한다. `Saver`는 **체크포인트 파일**(checkpoint file)인 이진 파일을 이용하여 모델의 매개변수를 저장하고 복원한다. 

텐서플로의 `Saver`는 `tf.train.Saver()`를 통해 사용할 수 있으며, `tf.train.Saver()`의 `.save()`메소드를 이용해 체크포인트 파일을 저장한다. 

10.1.1에서 구현한 Softmax Regression 모델을 텐서플로의 `Saver`를 이용해 저장하고 불러와 보도록 하자.


```python
import os
import tensorflow as tf

# MNIST data load
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x, test_x = train_x.reshape(-1, 28*28), test_x.reshape(-1, 28*28)
train_x, test_x = train_x/255., test_x/255.
train_y = tf.keras.utils.to_categorical(train_y, 10)
test_y = tf.keras.utils.to_categorical(test_y, 10)

inputs = tf.placeholder(tf.float32, [None, 28*28], name='inputs')
weights = tf.Variable(
        tf.truncated_normal(shape=[28*28, 10], stddev=0.01),
        name='weights')

logits = tf.matmul(inputs, weights)
labels = tf.placeholder(tf.float32, [None, 10])

# loss
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

# optimizer
train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# accuracy
correct_mask = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# Saver 정의
# 최근 7개의 체크포인트만 유지
saver = tf.train.Saver(max_to_keep=7, 
                       keep_checkpoint_every_n_hours=1)

# Hyper-Parameter
NUM_STEPS = 1000
MINIBATCH_SIZE = 100
TRAIN_SIZE = train_x.shape[0]
MODEL_PATH = './saved_model/'

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1, NUM_STEPS+1):
        # mini-batch using numpy
        batch_mask = np.random.choice(TRAIN_SIZE, MINIBATCH_SIZE)
        batch_x, batch_y = train_x[batch_mask], train_y[batch_mask]
        loss, _ = sess.run([cross_entropy, train_op], feed_dict={inputs:batch_x, 
                                                                 labels:batch_y})
        
        # 매 50 step 마다 학습된 가중치 저장
        if step % 50 == 0:
            saver.save(sess, 
                       os.path.join(MODEL_PATH, 'model_ckpt'),
                       global_step=step)
        
        if step % 200 == 0:
            test_acc = sess.run(accuracy, feed_dict={inputs: test_x, 
                                                     labels: test_y})
            print('Step: {:04d}, Loss: {:.5f}, Test_Acc: {:.5f}'.format(step, loss, test_acc))
            
    # 학습된 가중치를 res_weights에 할당하기
    res_weights = sess.run(weights)
print("최적화 완료")

# 출력결과
# Step: 0200, Loss: 0.81431, Test_Acc: 0.87860
# Step: 0400, Loss: 0.43764, Test_Acc: 0.89730
# Step: 0600, Loss: 0.37505, Test_Acc: 0.90310
# Step: 0800, Loss: 0.32732, Test_Acc: 0.91050
# Step: 1000, Loss: 0.38244, Test_Acc: 0.91260
# 최적화 완료
```




텐서플로의 `Saver`를 이용해 학습된 가중치를 저장하였으므로, 이번에는 `Saver.restore()`을 이용해 체크포인트를 복원하여 학습된 가중치를 모델에 할당해보자.


```python
tf.reset_default_graph()

# placeholder and variable
inputs = tf.placeholder(tf.float32, [None, 28*28], name='inputs')
weights = tf.Variable(
        tf.truncated_normal(shape=[28*28, 10], stddev=0.01),
        name='weights')

logits = tf.matmul(inputs, weights)
labels = tf.placeholder(tf.float32, [None, 10])

# accuracy
correct_mask = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# Saver
saver = tf.train.Saver()

# Test
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, os.path.join(MODEL_PATH, 'model_ckpt-1000'))
    acc = sess.run(accuracy, feed_dict={inputs: test_x,
                                        labels: test_y})
print("Accuarcy: {:.5f}".format(acc))
```

    INFO:tensorflow:Restoring parameters from ./saved_model/model_ckpt-1000
    Accuarcy: 0.91260




#### Saver를 이용해 그래프 복원하기

텐서플로 `Saver`의 장점은 연산 그래프를 저장해 다시 불러올 수 있다는 것이다. 위의 에제에서는 저장된 가중치 파일을 로드하여 그래프를 다시 구성한 뒤에 테스트를 수행했다. 텐서플로의 `Saver`는 기본적으로 체크포인트 파일을 저장할때 그래프 정보를 담고있는 `.meta` 파일도 같이 저장한다.

이렇게 저장된 `.meta` 파일을 텐서플로 `tf.train.import_meta_graph()`를 이용해 그래프를 불러온다. 아래의 예제코드는 `tf.train.import_meta_graph()`를 이용해 그래프를 불러와 테스트를 수행하는 코드이다. 학습단계에서 텐서플로의 컬렉션(collection)에 테스트 단계에 사용할 변수 `inputs, labels, accuracy`를추가하고 `Saver`의 `.export_meta_graph()`메소드의 인자 `collection_list`에 넣어준다.


```python
import os
import tensorflow as tf

# MNIST data load
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x, test_x = train_x.reshape(-1, 28*28), test_x.reshape(-1, 28*28)
train_x, test_x = train_x/255., test_x/255.
train_y = tf.keras.utils.to_categorical(train_y, 10)
test_y = tf.keras.utils.to_categorical(test_y, 10)

inputs = tf.placeholder(tf.float32, [None, 28*28], name='inputs')
weights = tf.Variable(
        tf.truncated_normal(shape=[28*28, 10], stddev=0.01),
        name='weights')

logits = tf.matmul(inputs, weights)
labels = tf.placeholder(tf.float32, [None, 10])

# loss
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

# optimizer
train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# accuracy
correct_mask = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# Saver 정의
# 최근 7개의 체크포인트만 유지
saver = tf.train.Saver(max_to_keep=7, 
                       keep_checkpoint_every_n_hours=1)

# 테스트에 쓰일 변수를 텐서플로 컬렉션에 저장
train_var = [inputs, labels, accuracy]
for var in train_var:
    tf.add_to_collection('train_var', var)

# Hyper-Parameter
NUM_STEPS = 1000
MINIBATCH_SIZE = 100
TRAIN_SIZE = train_x.shape[0]
MODEL_PATH = './saved_model/'

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1, NUM_STEPS+1):
        # mini-batch using numpy
        batch_mask = np.random.choice(TRAIN_SIZE, MINIBATCH_SIZE)
        batch_x, batch_y = train_x[batch_mask], train_y[batch_mask]
        loss, _ = sess.run([cross_entropy, train_op], feed_dict={inputs:batch_x, 
                                                                 labels:batch_y})
        
        # 매 50 step 마다 학습된 가중치 저장
        if step % 50 == 0:
            saver.export_meta_graph(os.path.join(MODEL_PATH, 'model_ckpt.meta'),
                                    collection_list=['train_var'])
            saver.save(sess, 
                       os.path.join(MODEL_PATH, 'model_ckpt'),
                       global_step=step)
        
        if step % 200 == 0:
            test_acc = sess.run(accuracy, feed_dict={inputs: test_x, 
                                                     labels: test_y})
            print('Step: {:04d}, Loss: {:.5f}, Test_Acc: {:.5f}'.format(step, loss, test_acc))
            
    # 학습된 가중치를 res_weights에 할당하기
    res_weights = sess.run(weights)
print("최적화 완료")

# 출력결과
# Step: 0200, Loss: 0.52200, Test_Acc: 0.87600
# Step: 0400, Loss: 0.49917, Test_Acc: 0.89760
# Step: 0600, Loss: 0.31210, Test_Acc: 0.90710
# Step: 0800, Loss: 0.27572, Test_Acc: 0.91070
# Step: 1000, Loss: 0.32306, Test_Acc: 0.91270
# 최적화 완료
```






이제 `tf.train.import_meta_graph()`를 이용해 저장한 그래프를 불러오고 `tf.get_collection()`을 통해 텐서플로 컬렉션에 저장한 변수들을 할당해준 후 테스트데이터로 테스트 해보자.


```python
tf.reset_default_graph()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Saver
    saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH, 'model_ckpt.meta')) 
    saver.restore(sess, os.path.join(MODEL_PATH, 'model_ckpt-1000'))
    inputs = tf.get_collection('train_var')[0]
    labels = tf.get_collection('train_var')[1]
    accuracy = tf.get_collection('train_var')[2]
    
    test_acc = sess.run(accuracy, feed_dict={inputs: test_x, 
                                             labels: test_y})

print("Accuarcy: {:.5f}".format(test_acc))
```

    INFO:tensorflow:Restoring parameters from ./saved_model/model_ckpt-1000
    Accuarcy: 0.91270




### 10.1.3 정리

텐서플로에서 학습한 가중치를 NumPy와 텐서플로의 Saver를 이용해 저장하고, 불러오는 방법에 대해 알아보았다. 여러번 학습을 시켜야 하거나, 학습된 모델을 바로 테스트 하는데 이러한 방법들을 이용해 편리하게 테스트할 수 있다. 
