import tensorflow as tf
import numpy as np

char_rdic = ['h', 'e', 'l', 'o'] #id -> char
char_dic = {W: i for i, W in enumerate(char_rdic)} # char -> id

# print(char_rdic)
# print(char_dic)

# input
x_data = np.array( ( [1,0,0,0], #h
	[0,1,0,0], #e
	[0,0,1,0], #l
	[0,0,1,0] ) , #l
	dtype='f')

# print(x_data)
sample = [ char_dic[c] for c in 'hello'] # to index
# print(sample)




# Configuration
# print(len(char_dic))
char_vocab_size = len(char_dic) # 4
rnn_size = char_vocab_size # 1 hot coding (one of 4)
time_step_size = 4 # 'hell' -> predict 'ello'
batch_size = 1 # one sample

# RNN model
# rnn_size = 4
rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
# print(rnn_cell)
# print(rnn_cell.state_size)

#how many cell = x_split !!!
state = tf.zeros([batch_size, rnn_cell.state_size]) # shape=(1, 4)
# X_split = tf.split(0, time_step_size, x_data)  # !!!
X_split = tf.split(x_data, time_step_size)  # !!!

print(X_split)
#print(type(rnn_cell))
#print(type(X_split))
#print(type(state))


# run cell
# outputs is Y
# outputs, state = tf.rnn.rnn(rnn_cell, X_split, state)
outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, X_split, state)

# Cost
# logis: list of 2D Tensors of shape [batch_size x num_decorder_symbols]. 예측값(출력값)
# targets: list of 1D batch-sized int32 Tensors of the same length as logits. 실제값 
# weights: list of 1D batch-sized float-Tensors of the same length as logits. 대부분 1
#print(outputs)
#print(tf.concat(outputs, 1))
logits = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])
# print(logits)
# print(sample[1:])
targets = tf.reshape(sample[1:], [-1])
# print(targets)
weights = tf.ones([time_step_size * batch_size])
# print(weights)

#loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# launch the graph in a session
with tf.Session() as sess:
	# you need to initialize all variables
	tf.initialize_all_variables().run()
	for i in range(120):
		sess.run(train_op)
		result = sess.run(tf.argmax(logits, 1))
		print(result, [char_rdic[t] for t in result])

