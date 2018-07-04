import tensorflow as tf

# generate Graph
a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[3., 4.], [1., 2.]])
c = tf.matmul(a, b)

with tf.Session(
    config=tf.ConfigProto(log_device_placement=True)) as sess:
    res = sess.run(c)
    
print(res)