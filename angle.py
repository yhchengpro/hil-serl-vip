from scipy.spatial.transform import Rotation as R

quat = [-0.03, 0.40, -0.02, 0.92]

euler = R.from_quat(quat).as_euler('xyz')

euler_str = ','.join(f"{v:.6f}" for v in euler)
print(euler_str)