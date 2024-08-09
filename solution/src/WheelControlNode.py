#!/usr/bin/env python3

import os
import rospy
import torch
import torch.nn as nn
import numpy as np
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Float32MultiArray

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class Actor(nn.Module):
    def __init__(self, action_dim=2, max_action=1.0, encoder=None):
        super(Actor, self).__init__()
        self.encoder = encoder 
        flat_size = 32 * 3 * 5
        self.actor = nn.Sequential(
            nn.Conv2d(128, 32, 3, stride=1, padding=1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Sigmoid()
        )
        self.max_action = max_action
    def forward(self, x):
        x = self.actor(x)
        print(f"Action shape from the actore network : {x.shape} , and the action is {x}`")
        return self.max_action * x

    def load_weights(self, weights):
        self.load_state_dict(torch.load(weights, map_location=device), strict=False)

class TD3Agent:
    def __init__(self, model_path):
        self.model = Actor()
        self.model.eval()
        self.model.load_weights(model_path)
        self.model.to(device)

    def get_action(self, latent_vector):
        # Convert latent vector to tensor
        latent_tensor = torch.from_numpy(latent_vector).float().unsqueeze(0).to(device)
        # Get action from the TD3 model
        with torch.no_grad():
            action = self.model(latent_tensor)
        return action.squeeze().cpu().numpy()

class WheelControlNode(DTROS):
    def __init__(self, node_name):

        # initialize the DTROS parent class
        super(WheelControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # static parameters
        try:
            self.vehicle_name = os.environ['VEHICLE_NAME']
        except KeyError:
            self.vehicle_name = 'marsbot'  # Default to 'marsbot' if VEHICLE_NAME is not set
            rospy.logwarn(f"VEHICLE_NAME not set. Using default: {self.vehicle_name}")
        #self.vehicle_name = os.environ['VEHICLE_NAME']#VEHICLE_NAME
        #vehicle_name = os.environ.get('VEHICLE_NAME','marsbot') # Default to 'marsbot' if VEHICLE_NAME is not set

        #vehicle_name = os.environ.get('VEHICLE_NAME')#, 'marsbot')
        wheels_topic = f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd"
        latent_topic = f"/{self.vehicle_name}/latent_vector"

        # Placeholder for accumulating data
        self.accumulated_data = []

        # construct publisher and subscriber
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        print(self._publisher,"************************************* publisher node working **********************************")
        self.sub = rospy.Subscriber(latent_topic, Float32MultiArray, self.callback)

        # load the TD3 model
        try:
            
            #print(f"Current working directory: {os.getcwd()}")
            #print(f"Contents of /code/catkin_ws/src/my-ros-project/weights/TD3-20240717-142333/: {os.listdir('/code/catkin_ws/src/my-ros-project/weights/TD3-20240717-142333/')}")
            
            #self._agent = TD3Agent('/code/catkin_ws/src/my-ros-project/weights/TD3-20240717-142333/td3.pth') # load the TD3 model
            self._agent = TD3Agent('code/catkin_ws/src/my-ros-project/weights/TD3-20240717-142333/td3.pth') # load the TD3 model
                       
            """current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(current_dir, '..', '..', '..', 'weights', 'TD3-20240717-142333', 'td3.pth')
            self._agent = TD3Agent(weights_path)
            print(f"TD3Agent initialized with weights: {weights_path}")"""
        except ValueError as e:
            rospy.logerr(str(e))
            rospy.signal_shutdown("Error initializing TD3Agent")  


    def callback(self, msg):

         # Convert message data to numpy array
        latent_vector = np.array(msg.data)
        print(latent_vector.size,"************************************* data from the camera **********************************")

        # get action from the TD3 agent
        action = self._agent.get_action(latent_vector)

        # form the wheels command message
        message = WheelsCmdStamped()
        message.vel_left = action[1]#*(1/2.2) # action[0]
        message.vel_right = action[0]#*(3/5) # action[1]

        # publish the message
        self._publisher.publish(message)

    def on_shutdown(self):
        stop = WheelsCmdStamped()
        stop.vel_left = 0
        stop.vel_right = 0
        self._publisher.publish(stop)

if __name__ == '__main__':
    try:
        # create the node
        node = WheelControlNode(node_name='wheel_control_node') # create a node with the name 'wheel_control_node'WheelControlNode
        # keep spinning
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
