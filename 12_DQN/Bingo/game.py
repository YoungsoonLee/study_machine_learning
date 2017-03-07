import numpy as np
import random

def checkBingo(array):
	win_number = 0

	for i in range(0, len(array)):
		if np.equal(array[i, :], 1).all():
			win_number = 1
		elif np.equal(array[i, :], 2).all():
			win_number = 2
		elif np.equal(array[:, i], 1).all():
			win_number = 1
		elif np.equal(array[:, i], 2).all():
			win_number = 2

	# check diagonal
	if np.equal(array.diagonal(), 1).all():
		win_number = 1
	elif np.equal(array.diagonal(), 2).all():
		win_number = 2
	elif np.equal(np.diag(np.fliplr(array)), 1).all():
		win_number = 1
	elif np.equal(np.diag(np.fliplr(array)), 2).all():
		win_number = 2
	elif array.all() > 0:
		win_number = 3
	return win_number

panel = np.zeros( (3,3) )
#panel = np.full((3, 3), 0)

loop = False
i = 0 
while not loop:
	print(panel)
	xy = input("input(x,y): ")
	xy = xy.replace(',','')

	x = int(xy[0])
	y = int(xy[1])

	if panel[x][y] != 0:
		print('already have a position')
	else:
		if i % 2 == 0:
			panel[x][y] = 1
		else:
			#cpu
			panel[x][y] = 2

	#check bingo
	result = checkBingo(panel)
	if result == 1:
		print('Bingo User')
		print(panel)
		break
	elif result == 2:
		print('Bingo Cpu')
		print(panel)
		break
	elif result == 3:
		print('Draw')
		print(panel)
		break
	else:
		i += 1		

	#loop = panel.all()