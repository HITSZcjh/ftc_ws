import rospy
from PPO import PPO
from nav_msgs.msg import Odometry

load_actor_model_path = "/home/jiao/rl_ws/test_uav/model/actor_model12-10-2023_11-19-48"
load_critic_model_path = "/home/jiao/rl_ws/test_uav/model/critic_model12-10-2023_11-19-48"
if __name__ == "__main__":
    ppo = PPO(None, None, None)
    ppo.load_model(load_actor_model_path, load_critic_model_path, 120)

    rospy.init_node("VisualizeModel_node", anonymous=True)

    rate = rospy.Rate(100)
    odometry_pub = rospy.Publisher("UAV_odometry", Odometry)

    odom = Odometry()
    while(not rospy.is_shutdown()):
        states = ppo.get_path()["states"]
        for i in range(states.shape[0]):
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "world"
            odom.pose.pose.position.x = states[i,0]
            odom.pose.pose.position.y = states[i,1]
            odom.pose.pose.position.z = states[i,2]
            odom.pose.pose.orientation.w = states[i,9]
            odom.pose.pose.orientation.x = states[i,10]
            odom.pose.pose.orientation.y = states[i,11]
            odom.pose.pose.orientation.z = states[i,12]
            odometry_pub.publish(odom)
            rate.sleep()
            
