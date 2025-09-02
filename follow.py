def follow():
    """
    Follow the lead robot arm
    """
    import argparse
    import time
    import numpy as np

    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
    from scipy.spatial.transform import Rotation as R

    parser = argparse.ArgumentParser(description="Switch control mode example")
    parser.add_argument(
        "--lead-url",
        type=str,
        default="localhost",
        help="URL to connect to the lead server",
    )
    parser.add_argument(
        "--follow-url",
        type=str,
        default="localhost",
        help="URL to connect to the follow server",
    )
    parser.add_argument(
        "--lead-port", type=int, default=50050, help="Server port for arm to lead"
    )
    parser.add_argument(
        "--follow-port", type=int, default=50051, help="Server port for arm to follow"
    )

    args = parser.parse_args()
    if args.lead_port == args.follow_port and args.lead_url == args.follow_url:
        raise ValueError(
            "Lead and follow port, lead and follow url cannot be the same at the same time."
        )

    with AIRBOTPlay(
        url=args.lead_url,
        port=args.lead_port,
    ) as robot1, AIRBOTPlay(
        url=args.follow_url,
        port=args.follow_port,
    ) as robot2:
        robot1.switch_mode(RobotMode.PLANNING_POS)
        if (
            sum(
                abs(i - j)
                for i, j in zip(robot1.get_joint_pos(), robot2.get_joint_pos())
            )
            > 0.1
        ):
            robot2.switch_mode(RobotMode.PLANNING_POS)
            robot2.move_to_joint_pos(robot1.get_joint_pos())
            time.sleep(1)
        print("Lead robot arm is ready to follow.")
        robot1.switch_mode(RobotMode.GRAVITY_COMP)
        robot2.switch_mode(RobotMode.SERVO_JOINT_POS)
        robot2.set_speed_profile(SpeedProfile.FAST)
        all_data = []
        try:
            while True:
                robot2.servo_joint_pos(robot1.get_joint_pos())
                robot2.servo_eef_pos(robot1.get_eef_pos() or [])
                data = robot1.get_end_pose()
                all_data.append(data)
                flattened = [d[0] + d[1] for d in all_data]
                arr = np.array(flattened)
                max_values = arr.max(axis=0).tolist()
                min_values = arr.min(axis=0).tolist()
                print("最大值：", max_values)
                print("最小值：", min_values)
                time.sleep(0.01)
        finally:
            robot1.switch_mode(RobotMode.PLANNING_POS)
            robot2.switch_mode(RobotMode.PLANNING_POS)
            robot2.set_speed_profile(SpeedProfile.DEFAULT)
            
if __name__ == "__main__":
    follow()
#[-0.028363730758428574, -0.5588275790214539, 1.0564393997192383, -1.6101375818252563, 1.3191606998443604, 1.5133827924728394]
#z=0.382 0.17
#y=+-0.18
#x 0.25 0.6

# 最大值： [0.3383518005361673, 0.14675747594653432, 0.250753162585558, 0.05236487763394989, 0.37251896571393994, 0.0940964758495676, 0.9663478416165716]
# 最小值： [0.23248433948148395, 0.027807754216531873, 0.16833476628283356, -0.1401183397853807, 0.2541582284306704, -0.027447127665458752, 0.9250380368734742]