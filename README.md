## System Integration
Self Driving Car Nanodegree Capstone

### Project Overview

This is the capstone project for the Self Driving Car Nanodegree. This project includes writing ROS nodes for perception, planning, and control. The code is tested in simulation, and the vehicle is expected to drive in its lane, stay below the maximum speed/acceleration limits, and stop for red lights.

### Code Overview

The three ROS nodes for this project include the waypoint updater, dbw (drive-by-wire controller), and the traffic light detector. Here is the system architecture diagram, which shows the nodes and topics for the perception, planning, and control subsystems:

![final-project-ros-graph-v2](https://user-images.githubusercontent.com/74683142/132248344-f22b6876-6021-49d1-bc94-2a78bb130bcb.png)

#### Waypoint Updater Node (Planning)

The waypoint updater code can be found [here](https://github.com/saulakh/self-driving-car-capstone/blob/main/ros/src/waypoint_updater/waypoint_updater.py). This node subscribes to:
- `/current_pose` for the car's current position
- `/base_waypoints` for the waypoints along the track
- `/traffic_waypoint` for the state of the next traffic light

Using the car's current position and the state of the next traffic light, this node publishes a set of waypoints ahead of the car's position to the `/final_waypoints` topic. If the light is green or there is no traffic light ahead, the `/base_waypoints` are used and the car will continue driving along the track at the target velocity. 

If there is a red light ahead, this node uses the distance between the car's current position and the waypoint of the stop line for the upcoming traffic light, then changes the target velocity at each waypoint to decelerate the car. When the light turns green, it also changes the target velocity at each waypoint to accelerate the car, and this information is published to the `final_waypoints` topic.

From there, the waypoint follower node subscribes to the `final_waypoints` topic, and publishes the target linear and angular velocity to the `twist_cmd` topic. The DBW node changes the steering, throttle, and brake commands to match the target linear and angular velocity, and the car follows the updated waypoints.

#### DBW (Drive By Wire) Node (Control)

The DBW node includes the [dbw_node.py](https://github.com/saulakh/self-driving-car-capstone/blob/main/ros/src/twist_controller/dbw_node.py) and [twist_controller.py](https://github.com/saulakh/self-driving-car-capstone/blob/main/ros/src/twist_controller/twist_controller.py) files. The DBW node subscribes to:
- `/vehicle/dbw_enabled` to check whether dbw is enabled
- `/twist_cmd` for the linear and angular velocity commands
- `/current_velocity` for the car's current velocity

Using the target linear and angular velocity commands from the waypoint updater, the `twist_controller.py` file calculates the steering, throttle, and brake commands:

- The `yaw_controller` uses the target angular velocity from the waypoint updater, and calculates the output steering commands
- The `low pass filter` smooths out noisy measurements, and compares the car's current velocity to the target linear velocity from the waypoint updater. It uses this velocity error and a timestep of 0.02s to get the acceleration required for the car
- When this error is positive, the `throttle_controller` implements a PID controller and calculates the throttle values for the required acceleration, using the change between the current and previous measured time. The brake value is set to zero in this case
- When this error is negative, the torque required to decelerate the car is used as the brake value, and the throttle is set to zero

The DBW node publishes these calculated steering, throttle, and brake commands to the `/vehicle/steering_cmd`, `/vehicle/throttle_cmd`, and `/vehicle/brake_cmd` topics.

#### Traffic Light Detector Node (Perception)

The traffic light detector node includes the [tl_detector.py](https://github.com/saulakh/self-driving-car-capstone/blob/main/ros/src/tl_detector/tl_detector.py) and [tl_classifier.py](https://github.com/saulakh/self-driving-car-capstone/blob/main/ros/src/tl_detector/light_classification/tl_classifier.py) files. The traffic light detector node subscribes to:
- `/base_waypoints` for the waypoints along the track
- `/image_color` for the current camera image in color
- `/current_pose` for the car's current position

This node uses the camera image and car's current position to check for red lights ahead, and publishes the light state to the `/traffic_waypoint` topic. The optional part of the project was to create a traffic light classifier, otherwise we had the option of using the given light_state from the simulator instead.

### Traffic Light Classifier

##### Object Detection

I decided to try the challenge version of the project and create my own traffic light classifier. I started with the mobilenet_coco_v1 model in tensorflow, since traffic lights were already included as one of the classes of objects.

##### Training Dataset

I saved camera images from the capstone simulation and created a .csv file with image paths and traffic light labels, similar to Udacity’s training data from the behavioral cloning project. In the tl_detector.py file from the capstone project, I started saving images from the get_light_state function:

![image](https://user-images.githubusercontent.com/74683142/138968555-da394425-4b87-436e-8f85-753ea39cdc05.png) ![image](https://user-images.githubusercontent.com/74683142/138968573-28c07f29-0f8b-44fa-8b95-9ede152ee9de.png) ![image](https://user-images.githubusercontent.com/74683142/138968659-9fba717d-c624-4347-a9b1-51e8103d9c35.png)

Next, I ran the object detection model through the folder of saved simulation images, and saved the cropped images into a folder as the training data for classification. This gave me a small dataset with 246 images of traffic lights. Here are some example images:

![image](https://user-images.githubusercontent.com/74683142/138967274-97c8b30a-24e8-4011-8af2-0948653e112f.png) ![image](https://user-images.githubusercontent.com/74683142/138967290-564a8e3c-f7fe-46b7-a283-b04a46453554.png) ![image](https://user-images.githubusercontent.com/74683142/138967307-ee93b989-be3b-4e9f-8d3f-b5d39dd960be.png) ![image](https://user-images.githubusercontent.com/74683142/138967322-8cdee898-cd7f-4eeb-9107-7d33858ba317.png) ![image](https://user-images.githubusercontent.com/74683142/138967332-50b78aa5-7127-4b88-9056-4ab3116b9f4d.png) ![image](https://user-images.githubusercontent.com/74683142/138967342-36863930-7099-4ac8-9041-c507154facc0.png) ![image](https://user-images.githubusercontent.com/74683142/138967366-10aae01e-70ee-4238-8721-6bab7a88db77.png) ![image](https://user-images.githubusercontent.com/74683142/138967381-ba05bfb6-247c-492a-abbd-ce52cf3607ca.png) ![image](https://user-images.githubusercontent.com/74683142/138967390-d40c1d6b-3e3f-4db9-abe5-42ccdf73c598.png)

##### Classifier Models

I started off practicing machine learning on my local computer, and created a separate repository for [traffic light classification](https://github.com/saulakh/traffic-light-classification). From there, I began to integrate the VGG16 keras model in this capstone project. I added a flatten layer and a dense layer with a softmax activation for 3 classes (red, yellow, green). On my local computer, this model worked well and correctly predicted the traffic light for 20 new simulation images that weren't included in the training or testing datasets. The object detection model was able to find and crop the traffic lights, and the keras model was able to successfully classify the light state.

![image](https://user-images.githubusercontent.com/74683142/137173033-1eedf0e9-b9f0-4868-a349-e5ac1bc4702d.png) ![image](https://user-images.githubusercontent.com/74683142/137173052-0f4a8f26-5374-4f72-b6ef-a15bdcf65832.png)

When training the model in the Udacity workspace, I had different results even with the same dataset and model architecture. I also wanted to minimize the computing time instead of using two machine learning models, so I decided not to use this keras model.

##### Image Processing

Next, I switched to using image processing in addition to object detection. Once I had a cropped traffic light image, I converted it to the HSV color space. I also isolated the v-channel and found pixels above 80% of the brightest value. I found the average y-value of these pixels and used the ratio of this y-value divided by the height of the image. Here is the initial camera image, the cropped light from object detection, the HSV image, and the isolated v-channel:

![image](https://user-images.githubusercontent.com/74683142/137419164-1febc380-a0f2-4a9b-bcb0-ac5e50ca174a.png) ![image](https://user-images.githubusercontent.com/74683142/137419176-30e1a7ef-5a89-40d0-abde-02ad628987c0.png) ![image](https://user-images.githubusercontent.com/74683142/137419191-affe228f-36e7-4ae8-80b0-4c99c0b91cb6.png) ![image](https://user-images.githubusercontent.com/74683142/137419203-8f5d1324-406d-4fc9-a775-c539ef544214.png)

From this light ratio, I started off assuming the top third was a red light, the middle third was a yellow light, and the bottom third meant a green light. This did not prove accurate, since some images had stray pixels that were bright enough and skewed the average y-value. Through trial and error, I ended up setting ratios below 0.5 as a red light, ratios between 0.5 and 0.6 as yellow, and anything over 0.6 as a green light. This worked well when the object detection model was able to detect traffic lights.

### Project Results

The car drove well as long as the coco model was able to detect traffic lights, and the image processing technique classified each light state correctly. I used print statements to compare the true light state to the predicted light state from the classifier, and for each discrepancy, the object detection model found 0 objects from the camera image.

If the object detection model did not see a traffic light, this threw an empty attribute error for the TL Classifier object and shut down the traffic light node. With more GPU hours for this project, I could have customized the COCO object detection model. The simulation traffic lights were different from real traffic lights, so I could have improved the accuracy by including simulation images in the training data. This issue happened a small percentage of the time, but I was running out of GPU hours to resolve this issue.

Here is an example image where the COCO object detection model did not see any traffic lights:

![image](https://user-images.githubusercontent.com/74683142/137419319-4a946f60-8bd9-4662-8828-50175be8f41c.png)

To get around this issue, I switched to using the provided light_state from the simulator for trafic light detection. After making this change, the car successfully completed the track without any incidents.

![image](https://user-images.githubusercontent.com/74683142/137419826-e55b1e43-2196-4227-9510-5f392f3e5046.png)

### Project Build Instructions

The simulator for this project can be downloaded [here](https://github.com/udacity/CarND-Capstone/releases), and the original project repository from Udacity can be found [here](https://github.com/udacity/CarND-Capstone) with the original README file and installation instructions.

1. Clone the project repository:

```
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies:

```
cd CarND-Capstone
pip install -r requirements.txt
```

3. Make and run styx:

```
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```

4. Launch the simulator.