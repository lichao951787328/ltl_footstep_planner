import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. 准备工作：创建包含 Z 坐标的示例数据文件 ---
def create_dummy_data_with_z():
    print("创建包含 Z 坐标的示例数据文件...")
    # os.makedirs("build", exist_ok=True)

    # env.data 内容 (x y z)
    # 两个矩形，Z 坐标设为 0.0
#     env_content = """
# 0.1 0.1 0.0
# 0.4 0.1 0.0
# 0.4 0.8 0.0
# 0.1 0.8 0.0
# 0.6 0.4 0.0
# 0.9 0.4 0.0
# 0.9 0.9 0.0
# 0.6 0.9 0.0
# """
#     # footsteps.data 内容 (x y z theta)
#     # Z 坐标可以变化，以证明它被正确忽略
#     footsteps_content = """
# 0.05 0.5 0.1 0.0
# 0.25 0.9 0.2 1.57
# 0.5 0.6 0.15 3.14
# 0.75 0.2 0.1 -1.57
# """
#     with open(os.path.join("build", "env.data"), "w") as f:
#         f.write(env_content.strip())
    
#     with open(os.path.join("build", "footsteps.data"), "w") as f:
#         f.write(footsteps_content.strip())
#     print("示例文件 'env.data' 和 'footsteps.data' 已创建。")

# # 调用函数创建文件
# create_dummy_data_with_z()

# --- 2. 读取数据文件 (关键修改在此处) ---
env_data_path = os.path.join("build", "env.data")
footsteps_data_path = os.path.join("build", "footsteps.data")

# 读取 env.data (障碍物矩形)
env_rectangles = []
try:
    with open(env_data_path, "r") as env_file:
        lines = [line.strip() for line in env_file if line.strip()]
        for i in range(0, len(lines), 4):
            # 提取每个顶点的 (x, y)，忽略 z
            rectangle_vertices_2d = []
            for line in lines[i:i+4]:
                parts = list(map(float, line.split()))
                if len(parts) >= 2: # 确保至少有x,y
                    rectangle_vertices_2d.append((parts[0], parts[1])) # 只取前两个元素
            
            if len(rectangle_vertices_2d) == 4:
                env_rectangles.append(rectangle_vertices_2d)
except FileNotFoundError:
    print(f"错误: 文件未找到 {env_data_path}")
    exit()

# 读取 footsteps.data (落脚点)
footsteps = []
try:
    with open(footsteps_data_path, "r") as footsteps_file:
        for line in footsteps_file.readlines():
            if not line.strip(): continue # 跳过空行
            parts = list(map(float, line.strip().split()))
            # 数据格式为 x, y, z, theta，我们需要 x, y, theta
            if len(parts) >= 4:
                x, y, z, theta = parts[0], parts[1], parts[2], parts[3]
                footsteps.append((x, y, theta)) # 忽略 z (parts[2])
except FileNotFoundError:
    print(f"错误: 文件未找到 {footsteps_data_path}")
    exit()

# --- 3. 开始绘图 (这部分代码无需改变) ---
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制环境中的所有矩形障碍物
print(f"正在绘制 {len(env_rectangles)} 个环境障碍物...")
for vertices in env_rectangles:
    polygon = patches.Polygon(
        vertices,
        facecolor='gray',
        edgecolor='black',
        alpha=0.6,
        linewidth=1.5
    )
    ax.add_patch(polygon)

# 绘制所有落脚点及其朝向
print(f"正在绘制 {len(footsteps)} 个落脚点...")
arrow_length = 0.05
for x, y, theta in footsteps:
    # 绘制落脚点中心
    ax.plot(x, y, 'bo', markersize=8, label='Footstep Position' if 'Footstep Position' not in ax.get_legend_handles_labels()[1] else "")
    
    # 计算并绘制朝向箭头
    dx = arrow_length * math.cos(theta)
    dy = arrow_length * math.sin(theta)
    ax.arrow(
        x, y, dx, dy,
        head_width=0.03,
        head_length=0.02,
        fc='red',
        ec='red',
        linewidth=1.5,
        label='Orientation (theta)' if 'Orientation (theta)' not in ax.get_legend_handles_labels()[1] else ""
    )

# --- 4. 美化和显示图形 (这部分代码无需改变) ---

# 自动计算坐标轴范围
all_x = []
all_y = []
for rect in env_rectangles:
    all_x.extend([v[0] for v in rect])
    all_y.extend([v[1] for v in rect])
all_x.extend([f[0] for f in footsteps])
all_y.extend([f[1] for f in footsteps])

if all_x and all_y:
    margin = 0.1
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

# 设置图形属性
ax.set_aspect('equal', adjustable='box')
plt.title("Environment Obstacles and Footstep Plan (2D Projection)")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 显示图形
plt.show()