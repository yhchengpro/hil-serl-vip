"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`
"""
from flask import Flask, request, jsonify
import numpy as np
import time
import threading
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile
from airbot_kdl import ArmKdl


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "127.0.0.1", "Ip address of the airbot play robot"
)
flags.DEFINE_integer(
    "robot_port", 50051, "gRPC port of the airbot plat robot"
)
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, 0, 0, 0, 1],
    "Target joint angles for the robot to reset to",
)
flags.DEFINE_string("flask_url", 
    "127.0.0.1",
    "URL for the flask server to run on."
)


class AirbotPlayServer():
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""

    def __init__(self, robot_ip, robot_port, reset_joint_target):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.reset_joint_target = reset_joint_target
        
        self.robot = AIRBOTArm(robot_ip, robot_port)
        self.robot.connect()
        eef = self.robot.get_product_info()["eef_types"][0]
        self.arm_kdl = ArmKdl(eef_type=eef)
        self.reset()
        

    
    def __del__(self):
        self.robot.disconnect()

    def reset(self):
        """Clears any errors"""
        self.robot.connect()
        self.robot.switch_mode(RobotMode.PLANNING_POS)
        self.robot.set_speed_profile(SpeedProfile.FAST)
        self.robot.move_to_joint_pos(self.reset_joint_target[:6])
        self.robot.move_eef_pos(self.reset_joint_target[6])

    def move(self, pose: list):
        """Moves to a pose: [x, y, z, qx, qy, qz, qw]"""
        assert len(pose) == 7
        self.robot.switch_mode(RobotMode.PLANNING_POS)
        trans = [pose[0], pose[1], pose[2]]
        orient = [pose[3], pose[4], pose[5], pose[6]]
        self.robot.move_cart_pose([trans, orient])

    def _set_currpos(self):
        self.pose = self.robot.get_end_pose()
        self.dq = self.robot.get_joint_vel()
        self.q = self.robot.get_joint_pos()
        self.jacobian = self.arm_kdl.jacobian(self.q)
        tau = self.robot.get_joint_eff()
        f = np.linalg.pinv(self.jacobian.T) @ tau
        self.force = np.array(f[:3])
        self.torque = np.array(f[3:])
        self.vel = self.jacobian @ self.dq
        self.eef_pos = self.robot.get_eef_pos()
        
    def _state_update_loop(self):
        """以 10ms 的频率更新一次状态"""
        while self._running:
            try:
                self._set_currpos()
            except Exception as e:
                print(f"[WARN] Failed to update robot state: {e}, trying to reconnect")
                self.robot.connect()
                time.sleep(1)
            time.sleep(0.01)

###############################################################################

def main(_):
    ROS_PKG_NAME = "serl_franka_controllers"

    ROBOT_IP = FLAGS.robot_ip
    ROBOT_PORT = FLAGS.robot_port
    FLASK_URI = FLAGS.flask_url
    RESET_JOINT_TARGET = FLAGS.reset_joint_target

    webapp = Flask(__name__)

    """Starts impedance controller"""
    robot_server = AirbotPlayServer(
        robot_ip=ROBOT_IP,
        robot_port=ROBOT_PORT,
        reset_joint_target=RESET_JOINT_TARGET,
    )
    
    # Route for pose in euler angles
    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        xyz = robot_server.pose[0]
        r = R.from_quat(robot_server.pose[1]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, r]).tolist()})

    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": np.array(robot_server.pose).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.array(robot_server.vel).tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.array(robot_server.force).tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.array(robot_server.torque).tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.array(robot_server.q).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.array(robot_server.dq).tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.array(robot_server.jacobian).tolist()})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": robot_server.eef_pos})

    # Route for Running Joint Reset
    @webapp.route("/reset", methods=["POST"])
    def joint_reset():
        robot_server.reset()
        return "Reset Joint"

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        print("open")
        robot_server.robot.move_eef_pos(1)
        return "Opened"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        print("close")
        robot_server.robot.move_eef_pos(0)
        return "Closed"

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        gripper_pos = request.json
        pos = np.clip(int(gripper_pos["gripper_pos"]), 0, 255)  # 0-255
        print(f"move gripper to {pos}")
        robot_server.robot.move_eef_pos(pos)
        return "Moved Gripper"

    # Route for Sending a pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pose_raw = np.array(request.json["arr"])
        robot_server.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
        joints = robot_server.arm_kdl.inverse_kinematics(construct_homogeneous_matrix(pose_raw), robot_server.q)[0]
        print("Moving to", pose, "joints:", joints)
        robot_server.robot.servo_joint_pos(joints)
        return "Moved"

    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify(
            {
                "pose": np.array(robot_server.pose[0] + robot_server.pose[1]).tolist(),
                "vel": np.array(robot_server.vel).tolist(),
                "force": np.array(robot_server.force).tolist(),
                "torque": np.array(robot_server.torque).tolist(),
                "q": np.array(robot_server.q).tolist(),
                "dq": np.array(robot_server.dq).tolist(),
                "jacobian": np.array(robot_server.jacobian).tolist(),
                "gripper_pos": np.array(robot_server.eef_pos).tolist(),
            }
        )

    webapp.run(host=FLAGS.flask_url)


if __name__ == "__main__":
    app.run(main)
