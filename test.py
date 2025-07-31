import numpy as np
from RNG_ros_interface import project_points_xyz_to_uv

# 手造一个 (x,y,z) = (0,0,0.5 m) 点 —— 光轴正前方 0.5 m
pts = np.array([[0.0, 0.0, 0.5]])
u, v = project_points_xyz_to_uv(pts)[0]
print(f"u={u:.1f}, v={v:.1f}")     # 应接近 (cx, cy) = (651, 357)