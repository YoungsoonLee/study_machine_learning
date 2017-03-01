import gym
from gym.envs.registration import register
#import sys, tty, termios

import os

# Windows
if os.name == 'nt':
    import msvcrt

# Posix (Linux, OS X)
else:
    import sys
    import termios
    import atexit
    from select import select


class KBHit:

    def __init__(self):
        '''Creates a KBHit object that you can call to do various keyboard things.
        '''

        if os.name == 'nt':
            pass

        else:

            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)

            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)


    def set_normal_term(self):
        ''' Resets to normal terminal.  On Windows this is a no-op.
        '''

        if os.name == 'nt':
            pass

        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)


    def getch(self):
        ''' Returns a keyboard character after kbhit() has been called.
            Should not be called in the same program as getarrow().
        '''

        s = ''

        if os.name == 'nt':
            return msvcrt.getch().decode('utf-8')

        else:
            return sys.stdin.read(1)


    def getarrow(self):
        ''' Returns an arrow-key code after kbhit() has been called. Codes are
        0 : up
        1 : right
        2 : down
        3 : left
        Should not be called in the same program as getch().
        '''

        if os.name == 'nt':
            msvcrt.getch() # skip 0xE0
            c = msvcrt.getch()
            vals = [72, 77, 80, 75]

        else:
            c = sys.stdin.read(3)[2]
            vals = [65, 67, 66, 68]

        return vals.index(ord(c.decode('utf-8')))


    def kbhit(self):
        ''' Returns True if keyboard character was hit, False otherwise.
        '''
        if os.name == 'nt':
            return msvcrt.kbhit()

        else:
            dr,dw,de = select([sys.stdin], [], [], 0)
            return dr != []

"""
class _Getch:
	def __call__(self):
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(3)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch

inkey = _Getch()
"""
# MACROS
# 0 : up
# 1 : right
# 2 : down
# 3 : left
UP = 3
DOWN = 1
RIGHT = 2
LEFT = 0

# key mapping
arrow_keys = {
	0: UP,
	1: RIGHT,
	2: DOWN,
	3: LEFT,
}


# Register FrozenLake with is_slippery False
register(
	id='FrozenLake-v3',
	entry_point='gym.envs.toy_text:FrozenLakeEnv',
	kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')
env.render() # show the initial board

kb = KBHit()

while True:
	# choose an action from keyboard
	if kb.kbhit():
		try:
			c = kb.getarrow()
			#print(c)
			action = arrow_keys[c]
			#print(action)

			state, reward, done, info = env.step(action)
			env.render() # show the board after action
			print('State: ', state, 'Action :', action, 'Reward :', reward, 'Info :', info)

			if done:
				print('Finished with reward: ', reward)
				break

		except ValueError:
			print('Not arrow keys')
			break
		# print(action)
kb.set_normal_term()