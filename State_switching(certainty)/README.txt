Welcome to the codebase for State Switching Deep Q Networks

This is the codebase for the paper accepted at Indian Controls Conference (2019)
- "Learning state switching for multi sensor integration"

1. Make sure ros/gazebo/sawyer simulation are all installed by following instructions in the getting started txt in the root of this directory.

2. Make sure keras is setup with theano or tensorflow

3. Run the file dqn_adv.py to start the training. 
	To train in simulation start the file after launching the sawyer simulation environment with sim flag. (Comments are provided as to which lines to change in dqn_adv.py when switching between sim and real). Also make sure table and cube is setup in the simulation.

	To train in real, first we need to pre-record a set of positions of the arm. Pre recorded files are provided in end_pos.obj and joint_record.obj. To create your own, use the code joint_recorder.py

4. Suggestive names are used for helper codes. sawyer.py is the RL environment for the training. Other codes are for controlling the arm and taking camera images.