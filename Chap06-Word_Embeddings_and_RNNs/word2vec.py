import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


####################
# Hyper Parameters #
####################
batch_size = 64
embedding_dimension = 5
negative_samples = 8
ABS_PATH = os.path.dirname(os.path.abspath('logs'))
LOG_DIR = ABS_PATH+'/logs/word2vec_intro'


digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
sentences = []

# 홀수 시퀀스/짝수 시퀀스 두 종류의 문장을 생성
for i in range(10000):
    rand_odd_ints = np.random.choice(range(1, 10, 2), size=3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    rand_even_ints = np.random.choice(range(2, 10, 2), size=3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))

# 생성된 문장 확인
print(sentences[0: 10])

# 단어를 인덱스에 매핑
word2index_map = {}
index = 0
for sent in sentences:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index+=1

index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)

print('word2index_map >>>', word2index_map)
print('index2word_map >>>', index2word_map)
print('vocabulary_size >>>', vocabulary_size)

# Skip-Gram 쌍(pair) 생성 (Window=1)
skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.lower().split()
    for i in range(1, len(tokenized_sent)-1):
        word_context_pair = [[word2index_map[tokenized_sent[i-1]],
                              word2index_map[tokenized_sent[i+1]]],
                             word2index_map[tokenized_sent[i]]]
        skip_gram_pairs.append([word_context_pair[1],
                                word_context_pair[0][0]])
        skip_gram_pairs.append([word_context_pair[1],
                                word_context_pair[0][1]])

print(skip_gram_pairs[0:10])

def get_skipgram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y

# mini-batch example
x_batch, y_batch = get_skipgram_batch(8)
print('x_batch :', x_batch)
print([index2word_map[word] for word in x_batch])
print('-'*30)
print('y_batch :', y_batch)
print([index2word_map[word[0]] for word in y_batch])

# 입력 데이터와 레이블
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

with tf.name_scope('embeddings'):
    embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_dimension],
                              -1.0, 1.0), name='embedding')
    # This is essentialy a lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Create variables for the NCE loss
nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_dimension],
                            stddev=1.0 / math.sqrt(embedding_dimension)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# NCE loss
loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_labels,
                   num_sampled=negative_samples, num_classes=vocabulary_size))
tf.summary.scalar('NCE_loss', loss)

# learning_rate decay
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.1,
                                           global_step=global_step,
                                           decay_steps=1000,
                                           decay_rate=0.95,
                                           staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 모든 요약 연산을 병합
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR,
                                         graph=tf.get_default_graph())
    saver = tf.train.Saver()
    
    with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))
            
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name
    # 임베딩을 메타데이터 파일과 연결
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    projector.visualize_embeddings(train_writer, config)
    
    tf.global_variables_initializer().run()
    
    for step in range(1000):
        x_batch, y_batch = get_skipgram_batch(batch_size)
        summary, _ = sess.run([merged, train_step], 
                              feed_dict={train_inputs: x_batch,
                                         train_labels: y_batch})
        train_writer.add_summary(summary, step)
        
        if step % 100 == 0:
            saver.save(sess, os.path.join(LOG_DIR, 'w2v_model.ckpt'), step)
            loss_value = sess.run(loss, feed_dict={train_inputs: x_batch,
                                                   train_labels: y_batch})
            print("Loss at %d: %.5f" % (step, loss_value))
            
    # 사용 전 임베딩 정규화
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)


ref_word = normalized_embeddings_matrix[word2index_map['one']]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
ff = np.argsort(cosine_dists)[::-1][1:10]
for f in ff:
    print('word: %s\tsim: %.5f' % (index2word_map[f], cosine_dists[f]))