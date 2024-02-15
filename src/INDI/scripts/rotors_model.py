import numpy as np

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from quadrotor_msgs.msg import ControlCommand
from mav_msgs.msg import Actuators

from matplotlib import pyplot as plt
import rospy
from uav_model import SimpleUAVModel

class RotorsUAVModel(object):
    def __init__(self, ts, delay_time=None) -> None:
        self.rotor_drag_coeff = 0.016  
        self.rotor_thrust_coeff = 8.54858e-06
        self.body_length = 0.17
        self.mass = 0.73
        self.g = 9.81
        self.inertia = np.array([[0.007, 0, 0], [0, 0.007, 0], [0, 0, 0.012]])
        self.max_rotors_speed = 838
        self.AllocationMatrix = np.array([[0, self.body_length, 0, -self.body_length],
                                [-self.body_length, 0, self.body_length, 0],
                                [self.rotor_drag_coeff, -self.rotor_drag_coeff, self.rotor_drag_coeff, -self.rotor_drag_coeff],
                                [1, 1, 1, 1]])
        self.AllocationMatrix_failed = np.array([[0, self.body_length, 0],
                                     [-self.body_length, 0, self.body_length],
                                     [1, 1, 1]])
        
        self.odometry_msg = None
        self.imu_msg = None
        self.motor_speed_msg = None

        rospy.Subscriber("/hummingbird/ground_truth/odometry", Odometry, self.odometry_callback)
        rospy.Subscriber("/hummingbird/ground_truth/imu", Imu, self.imu_callback)
        rospy.Subscriber("/hummingbird/motor_speed", Actuators, self.motor_speed_callback)
        
        self.cmd_pub = rospy.Publisher("/hummingbird/autopilot/control_command_input", ControlCommand, queue_size=1)
        self.cmd = ControlCommand()
        self.cmd.armed = True
        self.cmd.control_mode = ControlCommand.ROTOR_THRUSTS

        # self.cmd_pub = rospy.Publisher("/hummingbird/command/motor_speed", Actuators, queue_size=1)
        # self.cmd = Actuators()

        
        rate = rospy.Rate(100)

        while True:
            if self.odometry_msg is None or self.imu_msg is None or self.motor_speed_msg is None:
                rate.sleep()
            else:
                break
        
        for i in range(100):
            rate.sleep()

        self.delay_time = delay_time
        if self.delay_time is not None:
            self.ts = ts
            self.integrator = SimpleUAVModel(ts)
            num = int(delay_time/ts)
            self.f_target_list = [np.zeros(4) for _ in range(num)]

    def get_obs(self):
        p = np.array([self.odometry_msg.pose.pose.position.x, self.odometry_msg.pose.pose.position.y, self.odometry_msg.pose.pose.position.z])
        v_b = np.array([self.odometry_msg.twist.twist.linear.x, self.odometry_msg.twist.twist.linear.y, self.odometry_msg.twist.twist.linear.z])
        q = np.array([self.odometry_msg.pose.pose.orientation.w, self.odometry_msg.pose.pose.orientation.x, self.odometry_msg.pose.pose.orientation.y, self.odometry_msg.pose.pose.orientation.z])
        R = np.array([[1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
                      [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])],
                      [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]])
        v_w = R @ v_b
        w = np.array([self.odometry_msg.twist.twist.angular.x, self.odometry_msg.twist.twist.angular.y, self.odometry_msg.twist.twist.angular.z])
        f_real = np.array([self.motor_speed_msg.angular_velocities[0], self.motor_speed_msg.angular_velocities[1], self.motor_speed_msg.angular_velocities[2], self.motor_speed_msg.angular_velocities[3]])**2*self.rotor_thrust_coeff
        
        acc_B = np.array([self.imu_msg.linear_acceleration.x, self.imu_msg.linear_acceleration.y, self.imu_msg.linear_acceleration.z])

        if self.delay_time is not None:
            state = np.hstack((p, v_w, q, w, f_real))
            # 此处acc_B没有使用imu信息，而是采用名义模型的输出
            return self.integrator.predict(state, self.f_target_list, self.ts)

            # state = self.integrator.predict(state, self.f_target_list, self.ts)            
            # p, v_w, q, w, f_real = state[:3], state[3:6], state[6:10], state[10:13], state[13:]
            # R = np.array([[1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
            #           [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])],
            #           [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]])


        return np.hstack((p, v_w, q, w)), R, acc_B.reshape(-1,1), f_real

    def step(self, f_target):
        f_target = np.clip(f_target, 0, self.rotor_thrust_coeff*self.max_rotors_speed**2)

        if self.delay_time is not None:
            self.f_target_list.append(f_target)
            self.f_target_list.pop(0)

        self.cmd.header.stamp = rospy.Time.now()
        self.cmd.rotor_thrusts = f_target
        self.cmd_pub.publish(self.cmd)
        # self.cmd.angular_velocities = np.sqrt(f_target/self.rotor_thrust_coeff)
        # self.cmd_pub.publish(self.cmd)
        
    def odometry_callback(self, msg):
        self.odometry_msg = msg

    def imu_callback(self, msg):
        self.imu_msg = msg

    def motor_speed_callback(self, msg):
        self.motor_speed_msg = msg
if __name__ == "__main__":
    rospy.init_node("rotors_model")
    uav_model = RotorsUAVModel(0.01)

    rate = rospy.Rate(10)
    obs_list = []
    acc_B_list = []
    for i in range(100):
        obs, R, acc_B, f_real = uav_model.get_obs()
        obs_list.append(obs)
        acc_B_list.append(R@acc_B)
        print("obs",obs)
        print("R",R)
        print("f_real",f_real)
        print("acc_B",acc_B)
        rate.sleep()

    t = np.linspace(0,0.1*len(obs_list),len(obs_list))

    obs_list = np.array(obs_list)
    fig,axs = plt.subplots(obs_list.shape[1],1)
    fig.suptitle('obs')
    for i in range(obs_list.shape[1]):
        axs[i].plot(t, obs_list[:,i])

    acc_B_list = np.array(acc_B_list)
    fig,axs = plt.subplots(acc_B_list.shape[1],1)
    fig.suptitle('acc_B')
    for i in range(acc_B_list.shape[1]):
        axs[i].plot(t, acc_B_list[:,i])
    plt.show()
