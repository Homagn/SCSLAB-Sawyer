About this code:

Precise positioning of gripper over an object of interest using classic image processing
how? 
Learn a camera callibration matrix that can map center of the object in the image (in pixel location) to the actual world coordinate x,y positions relative to the robot.

For collecting callibration data, classic thresholding is used to detect center of a cube in an image (collected by both head and hand camera) and gazebo services are used to query the actual cube location.

In runtime, an object detector (eg. mask rcnn) can be used to detect bounding box for the object in the image and the center of the bounding box is mapped backt to the real world coordinates using learned camera callibration matrix.

After real world coordinates for center of the object are obtained, inverse kinematics is used to take the end effector to that position.

Setting up the simulator:

terminal: ./intera.sh sim
terminal: roslaunch sawyer_gazebo sawyer_world.launch electric_gripper:=true

If you have followed the sawyer setup instructions, then starting the simulator is very easy
(see the instructions to start sawyer simulator)

1. After starting simulator, drag and drop "wooden_cube_10cm" object from the gazebo models library
2. Drag and drop the wooden table ("cafe_table" probably)
3. run object_localization.py (check main function for options)



