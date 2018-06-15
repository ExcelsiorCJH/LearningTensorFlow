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


# Where to save TensorBoard model summaries
LOG_DIR = "./logs/RNN_with_summaries"

# Create placeholders for inputs, labels
_inputs = tf.placeholder(tf.float32, 
                         shape=[None, time_steps, element_size], 
                         name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')


# 요약(summaries)을 로깅(logging)하는 몇몇 연산을 추가하는 헬퍼 함수
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Weights and bias for input and hidden layer
with tf.name_scope('rnn_weights'):
    with tf.name_scope('W_x'):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope('W_h'):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope('Bias'):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)


def rnn_step(previous_hidden_state, x):
    current_hidden_state = tf.tanh(
            tf.matmul(previous_hidden_state, Wh) + 
            tf.matmul(x, Wx) + b_rnn)
    return current_hidden_state


# tf.scan() 함수로 입력값 처리
# input shape: (batch_size, time_steps, element_size)
processed_input = tf.transpose(_inputs, perm=[1, 0, 2])
# transposed input shape: (time_steps, batch_size, element_size)

initial_hidden = tf.zeros([batch_size, hidden_layer_size])
# time_steps에 따른 상태 벡터(state vector) 구하기
all_hidden_states = tf.scan(rnn_step, processed_input,
                            initializer=initial_hidden, name='states')


# 출력에 적용할 가중치
with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope('W_linear'):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], 
                                             mean=0, stddev=.01))
        variable_summaries(Wl)
    with tf.name_scope('Bias_linear'):
        bl = tf.Variable(tf.truncated_normal([num_classes], 
                                             mean=0, stddev=.01))
        variable_summaries(bl)
        
# 상태 벡터에 linear layer 적용
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
    # 시간에 따라 반복하면서 모든 RNN 결과에 get_linear_layer 적용
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
    # 최종 결과 = h_28
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)
    
with tf.name_scope('train'):
    # RMSPropOptimizer 사용
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
    
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
    tf.summary.scalar('accuracy', accuracy)
    
# 요약을 병합
merged = tf.summary.merge_all()

# Get a small test set
test_data = test_x[:batch_size].reshape([-1, time_steps, element_size])
test_label = test_y[:batch_size]

with tf.Session() as sess:
    # LOG_DIR에 텐세보드에서 사용할 요약을 기록
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', 
                                         graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', 
                                        graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    
        
    for step in range(10000):
        batch_x, batch_y = sess.run(next_batch)
        summary, _ = sess.run([merged, train_step], 
                              feed_dict={_inputs:batch_x['image'], y:batch_y})          
        # 요약 추가
        train_writer.add_summary(summary, step)
         
        if (step+1) % 1000 == 0:
            acc, loss = sess.run([accuracy, cross_entropy], 
                                 feed_dict={_inputs:batch_x['image'], y:batch_y})
            print("Iter: %04d\t" % (step+1), 
                  "MiniBatch Loss: {:.6f}\t".format(loss),
                  "Training Acc={:.5f}".format(acc))

        if (step+1) % 100 == 0:
            # MNIST 테스트 이미지에서 정확도를 계산해 요약에 추가
            summary, acc = sess.run([merged, accuracy], 
                                    feed_dict={_inputs: test_data, 
                                               y: test_label})
            test_writer.add_summary(summary, step)

                
    test_acc = sess.run(accuracy, feed_dict={_inputs: test_data,
                                             y: test_label})
    print("Test Accuracy:", test_acc)