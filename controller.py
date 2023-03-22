#! /usr/bin/env python
import rospy 
from lane_follower import LaneFollower
from laser_data import LaserData
from sign_recognizer import SignRecognizer
from object_tracker import ObjectTracker
from std_srvs.srv import Empty, EmptyResponse
from bot_message.srv import SetSpeed, SetSpeedResponse

class Controller(object):

  def __init__(self):
    self.forward_speed = 0.7 # Set the starting forward speed, DO NOT CHANGE
    self.lane_follower = LaneFollower(self.forward_speed)
    self.object_tracker = ObjectTracker()
    self.laser_data = LaserData(self.laser_callback)
    
    # Change the parameter depending on if you are collecting data or using a trained network to predict. 
    self.sign_recognizer = SignRecognizer(data_collection_mode=True, data_filename=None) 

    # Service definitions
    self.switch_left_service = rospy.Service("/switch_left", Empty, self.initiate_switch_left)
    self.switch_right_service = rospy.Service("/switch_right", Empty, self.initiate_switch_right)
    self.set_speed_service = rospy.Service("/set_speed", SetSpeed, self.set_speed_callback)

  # Command via terminal to call this function: rosservice call /set_speed "speed: 0.0"
  def set_speed_callback(self, speed_msg):
    # Max forward speed should be 1.2 m/s
    # Update forward speed
    # Update lane follower forward speed 
    velocity = max(0.0, min(speed_msg.speed, 1.2))
    self.forward_speed = velocity
    self.lane_follower.forward_speed = velocity
    
    return SetSpeedResponse()
  
  # Command via terminal to call this function: rosservice call /switch_left
  def initiate_switch_left(self, empty_msg):
    # Check line angles to see if you can switch lanes
    # Start the lane switching
    return EmptyResponse()
  
  # Command via terminal to call this function: rosservice call /switch_right
  def initiate_switch_right(self, empty_msg):
    # Check line angles to see if you can switch lanes
    # Start the lane switching
    return EmptyResponse()

  # Laser callback function, gets called at 10Hz
  def laser_callback(self, laser_msg):
    cartesian_data = self.laser_data.convert_to_cartesian(laser_msg)
    clusters = self.laser_data.cluster(cartesian_data)
    self.object_tracker.check_lanes(clusters)
    pass
   
if __name__ == "__main__":
  rospy.init_node("controller")
  controller = Controller()
  rospy.spin()
