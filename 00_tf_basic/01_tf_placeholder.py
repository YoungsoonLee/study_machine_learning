import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

# launch the default graph
with tf.Session() as sess:
	print(sess.run(add, feed_dict={a:2, b:3}))
	print(sess.run(mul, feed_dict={a:2, b:3}))