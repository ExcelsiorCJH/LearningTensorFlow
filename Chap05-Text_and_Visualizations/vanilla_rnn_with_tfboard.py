import tensorflow as tf

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

# dataset = tf.data.Dataset.from_tensor_slices(({"image": train_x}, train_y))
# dataset = dataset.shuffle(100000).repeat().batch(batch_size)
# iterator = dataset.make_one_shot_iterator()
# next_batch = iterator.get_next()

def next_batch(features, labels, batch_size):
    # An input function for training
        
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(({"image": features}, labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(buffer_size=100000).repeat().batch(batch_size)
    
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


#####################
# Define Parameters #
#####################
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

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