from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile
import numpy as np

def go_to_reset():
    reset_pose = [0.02872319146990776, -0.5373007655143738, 0.843009352684021, -1.6097872257232666, 1.1074951887130737, 1.5599446296691895]
    robot = AIRBOTArm("localhost", 50051)
    robot.connect()
    robot.switch_mode(RobotMode.PLANNING_POS)
    robot.servo_eef_pos(1)
    robot.servo_joint_pos(reset_pose)
    
def main():
    print("Starting the robot movement to reset position.")
    go_to_reset()
    print("Robot has reached the reset position.")

if __name__ == "__main__":
    main()