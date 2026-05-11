import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def on_policy_mc_epsilon_soft( environment, maxiters=5000, eps=0.3, gamma=0.99 ):
	"""
	Performs the on policy version of the every-visit MC control starting from the same state
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		eps: random value for the eps-greedy policy (probability of random action)
		gamma: gamma value, the discount factor for the Bellman equation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""

	p = [[1/environment.action_space for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	Q = [[0.0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	N = [[0   for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	print(f"Initial policy (eps={eps}):")
	print(f"{p}\n")
	print(f"Initial action-value function:")
	print(f"{Q}\n")
	print(f"Initial action counts:")
	print(f"{N}\n")
	for _ in range(maxiters):
		# generate episode using current eps-soft policy
		episode = environment.sample_episode(p)

		# compute returns backwards (every-visit)
		G = 0
		for state, action, reward in reversed(episode):
			G = reward + gamma * G
			N[state][action] += 1
			Q[state][action] += (G - Q[state][action]) / N[state][action]

			# update eps-soft policy for this state
			best = numpy.argmax(Q[state])
			for a in range(environment.action_space):
				if a == best:
					p[state][a] = 1 - eps + eps / environment.action_space
				else:
					p[state][a] = eps / environment.action_space

	deterministic_policy = [numpy.argmax(p[state]) for state in range(environment.observation_space)]
	return deterministic_policy


def on_policy_mc_exploring_starts( environment, maxiters=5000, eps=0.3, gamma=0.99 ):
	"""
	Performs the on policy version of the every-visit MC control starting from different states
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		eps: random value for the eps-greedy policy (probability of random action)
		gamma: gamma value, the discount factor for the Bellman equation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""

	p = [[1/environment.action_space for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	Q = [[0.0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	N = [[0   for _ in range(environment.action_space)] for _ in range(environment.observation_space)]

	for _ in range(maxiters):
		# exploring starts: force random first action then follow policy
		uniform = [[1/environment.action_space]*environment.action_space for _ in range(environment.observation_space)]
		episode = environment.sample_episode(uniform)

		# compute returns backwards (every-visit)
		G = 0
		for state, action, reward in reversed(episode):
			G = reward + gamma * G
			N[state][action] += 1
			Q[state][action] += (G - Q[state][action]) / N[state][action]

			# greedy policy update
			best = numpy.argmax(Q[state])
			p[state] = [1.0 if a == best else 0.0 for a in range(environment.action_space)]

	deterministic_policy = [numpy.argmax(p[state]) for state in range(environment.observation_space)]
	return deterministic_policy


def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the third lesson of the RL-Lab!   *" )
	print( "*            (Monte Carlo RL Methods)            *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()
	env.render()

	print( "\n3) MC On-Policy (with exploring starts)" )
	mc_policy = on_policy_mc_exploring_starts( env, maxiters=5000 )
	env.render_policy( mc_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(mc_policy) )
	
	print( "\n3) MC On-Policy (for epsilon-soft policies)" )
	mc_policy = on_policy_mc_epsilon_soft( env, maxiters=5000 )
	env.render_policy( mc_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(mc_policy) )
	

if __name__ == "__main__":
	main()
