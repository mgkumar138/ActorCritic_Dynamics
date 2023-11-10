# Dynamics of TD modulated Hebbian learning for spatial navigation to a single goal

## This repository contains the code:
1) Replicate the place cell based actor-critic architecture and learning rules in Foster, Dayan, Morris 2000
2) Replicate the experiment of target shift and re-learning setup in Zannone, Brzosko, Paulsen, Clopath 2018
3) One trial that does not end when the agent reaches the reward and with a longer max time to look at the TD curve convergence. 

## Task setup:
1) Initial training: The agent starts in the center of the square arena, and has to navigate to a single target in the top right corner. 
2) Re-learning: After the initial training, the reward location is shifted to the bottom left corner of the arena such that the agent has to un-learn its old policy and learn a new one. 

## Useful functions:
- Once an environment is initialized, the agent's trajectory for the last trial can be visualized using env.trajectory().
- Once an agent is initialized, its value and policy maps in terms of the weights from the place cells to the critic and actor can be visuzliased using agent.plot_maps(env), with the initialized environment parameter.
- The TD error can be accessed using agent.td and stored in a list outside the training loop.