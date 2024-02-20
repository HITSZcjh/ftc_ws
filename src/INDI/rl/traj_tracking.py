import rospy
from PPO import PPO
from nav_msgs.msg import Odometry,Path
from minimumsnap.msg import PolynomialTrajectory
import numpy as np
from geometry_msgs.msg import PoseStamped

coef = PolynomialTrajectory()
def CoefCallback(data):
    global coef
    coef = data

def get_visual_path(coefx, coefy, coefz, time, order):
    resource = 0.05
    path = Path()
    path.header.frame_id = "world"
    path.header.stamp = rospy.Time.now()
    total_time = np.sum(time)

    n = int(total_time/resource)
    t = np.linspace(0, total_time-resource, n)
    for j in range(n):
        dt = t[j]

        idx = 0
        while (dt > time[idx]):
            dt -= time[idx]
            idx += 1
        cur_poly_num = order + 1
        current_point = PoseStamped()
        current_point.header.frame_id = "world"
        current_point.pose.position.x = 0
        current_point.pose.position.y = 0
        current_point.pose.position.z = 0

        current_point.pose.orientation.x = 0
        current_point.pose.orientation.y = 0
        current_point.pose.orientation.z = 0
        current_point.pose.orientation.w = 1

        for i in range(cur_poly_num):
            current_point.pose.position.x += coefx[i, idx] * pow(dt, i)
            current_point.pose.position.y += coefy[i, idx] * pow(dt, i)
            current_point.pose.position.z += coefz[i, idx] * pow(dt, i)

        path.poses.append(current_point)
    return path


load_actor_model_path = "/home/jiao/rl_ws/test_uav/model/actor_model12-10-2023_11-19-48"
load_critic_model_path = "/home/jiao/rl_ws/test_uav/model/critic_model12-10-2023_11-19-48"
if __name__ == "__main__":
    rospy.init_node("VisualizeModel_node", anonymous=True)
    
    ppo = PPO(None, None, None)
    ppo.load_model(load_actor_model_path, load_critic_model_path, 120)

    rate = rospy.Rate(100)
    odometry_pub = rospy.Publisher("UAV_odometry", Odometry)
    way_point_pub = rospy.Publisher("waypoints", Path)
    desire_path_pub = rospy.Publisher("desire_path", Path)
    real_path_pub = rospy.Publisher("real_path", Path)
    coef_sub = rospy.Subscriber("polynomial_traj_coef",
                            PolynomialTrajectory, CoefCallback, queue_size=1)


    odom = Odometry()
    odom.header.stamp = rospy.Time.now()
    odom.header.frame_id = "world"
    odom.pose.pose.position.x = 0
    odom.pose.pose.position.y = 0
    odom.pose.pose.position.z = 3
    odom.pose.pose.orientation.w = 1
    odom.pose.pose.orientation.x = 0
    odom.pose.pose.orientation.y = 0
    odom.pose.pose.orientation.z = 0
    odometry_pub.publish(odom)
    rate.sleep()

    point = np.array([[0, 10, 3],[-7.5, 7.5, 3], [-3, 0, 3], [-7.5,-7.5, 3], [0,-10,3], [15, 0, 3]])
    way_point = Path()
    for i in range(point.shape[0]):
        way_point.header.stamp = rospy.Time.now()
        way_point.header.frame_id = "world"
        current_point = PoseStamped()
        current_point.header.frame_id = "world"
        current_point.pose.position.x = point[i][0]
        current_point.pose.position.y = point[i][1]
        current_point.pose.position.z = point[i][2]

        current_point.pose.orientation.x = 0
        current_point.pose.orientation.y = 0
        current_point.pose.orientation.z = 0
        current_point.pose.orientation.w = 1

        way_point.poses.append(current_point)
    way_point_pub.publish(way_point)

    while (len(coef.time) == 0):
        way_point_pub.publish(way_point)
        rate.sleep()

    _n_segment = len(coef.time)
    _order = coef.order
    _coefx = np.zeros((_order+1, _n_segment))
    _coefy = np.zeros((_order+1, _n_segment))
    _coefz = np.zeros((_order+1, _n_segment))
    _time = np.zeros(_n_segment)
    _start_time = _final_time = coef.header.stamp.to_sec()
    for i in range(_n_segment):
        _time[i] = coef.time[i]
        _final_time += _time[i]

    shift = 0
    for i in range(_n_segment):
        for j in range(_order+1):
            _coefx[j, i] = coef.coef_x[shift+j]
            _coefy[j, i] = coef.coef_y[shift+j]
            _coefz[j, i] = coef.coef_z[shift+j]
        shift += _order+1

    desire_path = get_visual_path(_coefx, _coefy, _coefz, _time, _order)

    desire_path_pub.publish(desire_path)
    rate.sleep()

    real_path = Path()

    _start_time = _final_time = rospy.Time.now().to_sec()
    for i in range(_n_segment):
        _final_time += _time[i]
    
    state = np.zeros(ppo.env.state_dim)
    state[0] = 0
    state[1] = 0
    state[2] = 3
    state[9] = 1

    while(not rospy.is_shutdown()):
        time_now = rospy.Time.now().to_sec()

        t = time_now - _start_time
        if(t>np.sum(_time)-0.0001):
            t = np.sum(_time)-0.0001
        idx = 0            
        while (t > _time[idx]):
            t -= _time[idx]
            idx += 1
        desire_pose = np.zeros(3)
        for j in range(_order+1):
            desire_pose[0] += _coefx[j, idx] * pow(t, j)
            desire_pose[1] += _coefy[j, idx] * pow(t, j)
            desire_pose[2] += _coefz[j, idx] * pow(t, j)

        state_copy = state.copy()
        state_copy[0:3] = state_copy[0:3] - desire_pose + np.array([0,0,3])
        print(state_copy)
        print(state)
        action = ppo.actor.get_action(state_copy)[0]
        state = ppo.env.integrate_only(state, action)

        odom.pose.pose.position.x = state[0]
        odom.pose.pose.position.y = state[1]
        odom.pose.pose.position.z = state[2]
        odom.pose.pose.orientation.w = state[9]
        odom.pose.pose.orientation.x = state[10]
        odom.pose.pose.orientation.y = state[11]
        odom.pose.pose.orientation.z = state[12]
        odometry_pub.publish(odom)

        real_path.header.stamp = rospy.Time.now()
        real_path.header.frame_id = "world"
        current_point = PoseStamped()
        current_point.header.frame_id = "world"
        current_point.pose.position.x = state[0]
        current_point.pose.position.y = state[1]
        current_point.pose.position.z = state[2]

        current_point.pose.orientation.x = 0
        current_point.pose.orientation.y = 0
        current_point.pose.orientation.z = 0
        current_point.pose.orientation.w = 1

        real_path.poses.append(current_point)
        real_path_pub.publish(real_path)
        desire_path_pub.publish(desire_path)
        time_record = rospy.Time.now().to_sec() - time_now
        print("estimation time is {}".format(time_record))
        rate.sleep()