import rospy
import numpy as np
from std_msgs.msg import Float64
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from sensor_msgs.msg import Image
import sys, select, os
import roslib
if os.name == 'nt':
  import msvcrt
else:
  import tty, termios

import mpc_tool as mpc

roslib.load_manifest('dual_gazebo')

g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

def qua2eular(x,y,z,w):

    q_x = x
    q_y = y
    q_z = z
    q_w = w

    t0 = +2.0 * (q_w * q_x + q_y * q_z)
    t1 = +1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (q_w * q_y - q_z * q_x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (q_w * q_z + q_x * q_y)
    t4 = +1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians


def move_mecanum(data):
    # start publisher of cmd_vel to control mecanum

    linear, angular = data

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    twist = Twist()

    twist.linear.x = linear

    pub.publish(twist)

    print(twist)

def get_model_status():

    robot_state = g_get_state(model_name = 'dual_robot')

    robot = Pose()
    robot.position.x = float(robot_state.pose.position.x)
    robot.position.y = float(robot_state.pose.position.y)
    robot.position.z = float(robot_state.pose.position.z)
    robot.orientation.x = float(robot_state.pose.orientation.x)
    robot.orientation.y = float(robot_state.pose.orientation.y)
    robot.orientation.z = float(robot_state.pose.orientation.z)
    robot.orientation.w = float(robot_state.pose.orientation.w)

    robot_vel = Twist()

    robot_vel.linear.x = float(robot_state.twist.linear.x)
    robot_vel.linear.y = float(robot_state.twist.linear.y)
    robot_vel.linear.z = float(robot_state.twist.linear.z)

    robot_vel.angular.x = float(robot_state.twist.angular.x)
    robot_vel.angular.y = float(robot_state.twist.angular.y)
    robot_vel.angular.z = float(robot_state.twist.angular.z)

    return robot, robot_vel

if __name__ == '__main__':
    rospy.init_node('mecanum_key')


    robot, robot_vel = get_model_status()

    roll_x, pitch_y, yaw_z = qua2eular(robot.orientation.x, robot.orientation.y, robot.orientation.z, robot.orientation.w)

    mpc_model, _, u0 = mpc.set_model(pitch_y)
    
    x0 = np.array([[robot.position.x],
                    [pitch_y],
                    [robot_vel.linear.x],
                    [robot_vel.angular.y]])

    linear = [0, 0, 0]
    angular = [0, 0, 0]

    while(1):
        u0 = mpc_model.make_step(x0)

        input_vel = u0 / 21 
        
        move_mecanum([input_vel, 0])

        robot, robot_vel = get_model_status()
        roll_x, pitch_y, yaw_z = qua2eular(robot.orientation.x, robot.orientation.y, robot.orientation.z, robot.orientation.w)
        x0 = np.array([[robot.position.x],
                    [pitch_y],
                    [robot_vel.linear.x],
                    [robot_vel.angular.y]])