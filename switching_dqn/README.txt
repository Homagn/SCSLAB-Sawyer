Installation requirements:
python -m pip install --user theano
python -m pip install --user tensorflow=1.14.0
python -m pip install --user keras==2.0.4

cd ros_ws
source /opt/ros/kinetic/setup.bash
./intera.sh  (or ./intera.sh sim)
(to check connection to robot- ping sawyer.local)

Setting up the simulator:

If you have followed the sawyer setup instructions, then starting the simulator is very easy
(see the instructions to start sawyer simulator)

1. After starting simulator, drag and drop "wooden_cube_10cm" object from the gazebo models library
2. Drag and drop the wooden table ("cafe_table" probably)
3. Place wooden cube on wooden table
4. Start training code after passing appropriate arguments

Setting up the real robot:
1. Set the table up with tapes and markers for consistency
2. Record joint positions using joint_recorder.py (can also write a better code to do 3 step/4 step cube re-placement)
3. Get the special 3d printed training cube from the drawyer under my table

