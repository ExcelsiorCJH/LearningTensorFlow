import tensorflow as tf


#####################
# Define Parameters #
#####################
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128


# MNIST 데이터 불러오기 위한 함수 정의
def mnist_load():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    # Train set
    train_x = train_x.astype('float32') / 255.
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)
    # Test set
    test_x = test_x.astype('float32') / 255.
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=10)
    
    return (train_x, train_y), (test_x, test_y)

# MNIST 데이터 불러오기
(train_x, train_y), (test_x, test_y) = mnist_load()

dataset = tf.data.Dataset.from_tensor_slices(({"image": train_x}, train_y))
dataset = dataset.shuffle(100000).repeat().batch(batch_size)
iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

# 1) Create placeholders for inputs, labels
_inputs = tf.placeholder(tf.float32, shape=[None, time_steps,
                                            element_size], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='inputs')

# 2) RNN Model
# Tensorflow built-in functions
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_layer_size)
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                     mean=0, stddev=.01))
bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))

def get_linear_layer(vector):
    return tf.matmul(vector, Wl) + bl

last_rnn_output = outputs[:,-1,:]
final_output = get_linear_layer(last_rnn_output)

# 3) Loss function
cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output, labels=y))
# 4) Optimizer
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

# 5) accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

# 6) Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Get a small test set
    test_data = test_x[:batch_size].reshape([-1, time_steps, element_size])
    test_label = test_y[:batch_size]
    
    for step in range(10000):
        batch_x, batch_y = sess.run(next_batch)
        
        sess.run(train_step, feed_dict={_inputs:batch_x['image'], 
                                        y:batch_y})          
        
         
        if (step+1) % 1000 == 0:
            acc, loss = sess.run([accuracy, cross_entropy], 
                                 feed_dict={_inputs:batch_x['image'], y:batch_y})
            print("Iter: %04d\t" % (step+1), 
                  "MiniBatch Loss: {:.6f}\t".format(loss),
                  "Training Acc={:.5f}".format(acc))

        if (step+1) % 100 == 0:
            # MNIST 테스트 이미지에서 정확도를 계산해 요약에 추가
            acc = sess.run(accuracy, feed_dict={_inputs: test_data, 
                                                y: test_label})

                
    test_acc = sess.run(accuracy, feed_dict={_inputs: test_data,
                                             y: test_label})
    print("Test Accuracy:", test_acc)