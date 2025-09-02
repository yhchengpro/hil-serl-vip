from airbot_py.arm import AIRBOTPlay, RobotMode
from show_rlpd import init, task
from experiments.grasp_bread.config import TrainConfig as GraspBreadTrainingConfig
from experiments.move_bread.config import TrainConfig as MoveBreadTrainConfig
from experiments.press_switch.config import TrainConfig as PressSwitchTrainConfig
import time

def main():
    
    sampling_rng1, agent1, env1 = init(configname = GraspBreadTrainingConfig, start_step = 75000, checkpoint_path = "/home/kkk/workspace/hil-serl/examples/experiments/grasp_bread/first_run")
    task(sampling_rng1, agent1, env1)
    env1.close()
    
    print("env1_closed")

    with AIRBOTPlay(
        url = "localhost",
        port = 50051,
    )as robot:
        robot.switch_mode(RobotMode.PLANNING_POS)
        robot.move_eef_pos(0)
    
    sampling_rng2, agent2, env2 = init(configname = MoveBreadTrainConfig, start_step = 65000, checkpoint_path = "/home/kkk/workspace/hil-serl/examples/experiments/move_bread/first_run")
    task(sampling_rng2, agent2, env2)
    env2.close()

    with AIRBOTPlay(
        url = "localhost",
        port = 50051,
    )as robot:
        robot.switch_mode(RobotMode.PLANNING_POS)
        pose = robot.get_end_pose()
        print(f"pose={pose}")
        pose[0][2] -= 0.08
        robot.move_to_cart_pose(pose)
        print("moved!!!")
        robot.move_eef_pos(1)
        print("eef!!!")
        pose = robot.get_end_pose()
        pose[0][2] += 0.12
        robot.move_to_cart_pose(pose)
        
    sampling_rng3, agent3, env3 = init(configname = PressSwitchTrainConfig, start_step = 40000, checkpoint_path = "/home/kkk/workspace/hil-serl/examples/experiments/press_switch/first_run")
    task(sampling_rng3, agent3, env3)
    env3.close()
    
    with AIRBOTPlay(
        url = "localhost",
        port = 50051,
    )as robot:
        pose = robot.get_end_pose()
        pose[0][2] -= 0.06
        robot.switch_mode(RobotMode.PLANNING_POS)
        robot.move_to_cart_pose(pose)

if __name__ == "__main__":
    main()