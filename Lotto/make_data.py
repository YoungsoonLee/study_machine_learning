import random


data = []
base_number = [x for x in range(1,50)]
output = ''

for i in range(0,3):
	output += str(1)+'\t' # bias
	rnd = random.sample(base_number, 8)
	for i in range(0, len(rnd)):
		output += str(rnd[i])+'\t'
		if i == 7:
			output += '0'+'\n'

# print(output)

# test y data
y = [ [5, 17, 19, 25, 31, 38, 46, 4, 1], [8, 27, 28, 29, 31, 32, 35, 11, 1], [5, 15, 25, 38, 42, 46, 47, 26, 1]]

for j in range(0,3):
	output += str(1)+'\t'
	for i in range(0, 9):			
			if i == 8:
				output += str(y[j][i])
				output += '\n'
			else:
				output += str(y[j][i])+'\t'

# print(output)

p = '{0}'.format(output)
with open("train.txt", "w") as f:
	f.write(p)
	