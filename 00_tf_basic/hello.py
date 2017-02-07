import tensorflow as tf

# hello is operation~!
# everythins is operation!
hello = tf.constant('Hello Tensorflow')

# start tf session
sess = tf.Session()

print(sess.run(hello))