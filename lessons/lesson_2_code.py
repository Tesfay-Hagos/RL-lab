import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def random_dangerous_grid_world( environment ):
	"""
	Performs a random trajectory on the given Dangerous Grid World environment 
	
	Args:
		environment: OpenAI Gym environment
		
	Returns:
		trajectory: an array containing the sequence of states visited by the agent
	"""
	trajectory = []
	environment.robot_state = environment.start_state
	trajectory.append( environment.robot_state )
	for step in range(10):
		a = numpy.random.randint( 0, environment.action_space )
		new_state = environment.sample( a )
		trajectory.append( new_state )
		if environment.is_terminal( new_state ): break
		environment.robot_state = new_state
	return trajectory


class RecyclingRobot():
	"""
	Class that implements the environment Recycling Robot of the book: 'Reinforcement
	Learning: an introduction, Sutton & Barto'. Example 3.3 page 52 (second edition).
		
	Attributes
	----------
		observation_space : int
			define the number of possible actions of the environment
		action_space: int
			define the number of possible states of the environment
		actions: dict
			a dictionary that translate the 'action code' in human languages
		states: dict
			a dictionary that translate the 'state code' in human languages
		
	Methods
	-------
		reset( self )
			method that reset the environment to an initial state; returns the state
		step( self, action )
			method that perform the action given in input, computes the next state and the reward; returns 
			next_state and reward
		render( self )
			method that print the internal state of the environment
	"""


	def __init__( self ):
		# Loading the default parameters
		self.alfa = 0.7
		self.beta = 0.7
		self.r_search = 0.5
		self.r_wait = 0.2

		# Define the state and action space
		self.states = {0: "high", 1: "low"}
		self.observation_space = len(self.states)	
		self.actions = {0: "search", 1: "wait", 2: "recharge"}
		self.action_space = len(self.actions)

		# Initialize the state
		self.state = None


	def reset( self ):
		self.state = 0  # start with high battery
		return self.state


	def step( self, action ):

		# Default reward
		reward = 0

		# If current state is HIGH battery
		if self.state == 0:  # high

			if action == 0:  # search
				reward = self.r_search
				# With probability alfa, stay high; otherwise go low
				if numpy.random.rand() < self.alfa:
					next_state = 0  # stay high
				else:
					next_state = 1  # go low

			elif action == 1:  # wait
				reward = self.r_wait
				next_state = 0  # waiting keeps battery high

			elif action == 2:  # recharge
				reward = 0.0
				next_state = 0  # already high, stays high

			else:
				raise ValueError("Invalid action")

		# If current state is LOW battery
		elif self.state == 1:  # low

			if action == 0:  # search
				reward = self.r_search
				# With probability beta, stay low; otherwise 'fail' → go high or terminal
				if numpy.random.rand() < self.beta:
					next_state = 1  # stay low
				else:
					# In the book, failure might be modeled differently; here we send it to high
					next_state = 0  # e.g., forced recharge or reset

			elif action == 1:  # wait
				reward = self.r_wait
				next_state = 1  # waiting keeps battery low

			elif action == 2:  # recharge
				reward = 0.0
				next_state = 0  # recharge → high

			else:
				raise ValueError("Invalid action")

		else:
			raise ValueError("Invalid state")

		# Update internal state
		self.state = next_state

		# No terminal state in this simple version
		done = False
		info = None

		return self.state, reward, done, info



	def render( self ):
		print( f"Current state: '{self.states[self.state]}'" )
		return True


def main():
	print( "\n************************************************" )
	print( "*  Welcome to the first lesson of the RL-Lab!  *" )
	print( "*             (MDP and Environments)           *" )
	print( "************************************************" )

	print( "\nA) Random Policy on Dangerous Grid World:" )
	env = GridWorld()
	env.render()
	random_trajectory = random_dangerous_grid_world( env )
	print( "\nRandom trajectory generated:", random_trajectory )


	print( "\nB) Custom Environment: Recycling Robot" )
	env = RecyclingRobot()
	state = env.reset()
	ep_reward = 0
	
	for step in range(10):
		a = numpy.random.randint( 0, env.action_space )
		new_state, r, _, _ = env.step( a )
		ep_reward += r
		print( f"\tFrom state '{env.states[state]}' selected action '{env.actions[a]}': \t total reward: {ep_reward:1.1f}" )
		state = new_state


if __name__ == "__main__":
	main()
