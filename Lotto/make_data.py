import random
from numpy import genfromtxt
import tensorflow as tf
import numpy as np


output = ''
my_data = genfromtxt('LOTTOMAX.csv', delimiter=',', skip_header=1, dtype='int16')
y = my_data[:, 4:]
# print(y)

# make x_data
base_number = [x for x in range(1,50)]
for i in range(0,len(y)):
	output += str(1)+'\t' # bias
	rnd = random.sample(base_number, 7)
	for i in range(0, len(rnd)):
		output += str(rnd[i])+'\t'
		if i == 6:
			output += '0'+'\n'

# make y_data
for j in range(0,len(y)):
	output += str(1)+'\t' # bias
	for i in range(0, 7):			
			if i == 6:
				output += str(y[j][i])+'\t'+'1'
				output += '\n'
			else:
				output += str(y[j][i])+'\t'

#print(output)

p = '{0}'.format(output)
with open("train.txt", "w") as f:
	f.write(p)
