import rospy
from sensor_msgs.msg import LaserScan
from laser_data import LaserData


def callback_function(laser_msg):
    global laser_data
    cartesian_data = laser_data.convert_to_cartesian(laser_msg)
    clusters = laser_data.cluster(cartesian_data)
    laser_data.plot_clusters(clusters)


if __name__ == "__main__":
  rospy.init_node("plot_clusters")
  laser_data = LaserData(callback_function)
  
  rospy.spin()