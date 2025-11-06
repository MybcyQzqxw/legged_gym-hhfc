from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgGAIL

# 参考轨迹文件路径
TRAJECTORY_FILE = f"{LEGGED_GYM_ROOT_DIR}/resources/trajectory/humanoid/standup01.dat"
# 状态维度 (欧拉角3 + 角速度3 + 关节位置12 + 关节速度12)
NUM_STATES = 30


class HhfcRobotCfg(LeggedRobotCfg):
    """Hhfc机器人的环境配置类

    继承自LeggedRobotCfg基类,定义了Hhfc人形机器人站立模仿任务的所有配置参数。
    """

    seed = 42  # 随机种子,用于结果复现

    class env(LeggedRobotCfg.env):
        """环境基本参数配置"""

        frame_stack = 10  # actor 网络保留 10 帧上下文 (捕捉时序信息)
        c_frame_stack = 3  # critic 网络保留 3 帧上下文
        # commands(3)+dof_pos(12)+dof_vel(12)+
        # actions(12)+base_ang_vel(3)+gravity(3)=45
        num_single_obs = 45  # 单帧观测维度
        num_observations = int(
            frame_stack * num_single_obs
        )  # 实际输入到actor网络的观测维度 (450)
        # 单帧的特权观测,即训练时 critic 网络才能使用的额外信息
        # privileged obs + base height -> 81
        single_num_privileged_obs = 81  # 单帧特权观测维度
        num_privileged_obs = int(
            c_frame_stack * single_num_privileged_obs
        )  # critic网络的特权观测维度 (243)
        num_actions = 12  # 动作维度 (12个关节)
        num_envs = 8192  # 并行环境数量 (越多训练越快,但需要更多GPU内存)
        episode_length_s = 20  # 每个 episode 最长 20 秒 (仿真时间)
        is_amp = True  # 使用 AMP (Adversarial Motion Priors) 算法
        num_states = NUM_STATES  # 状态维度 (用于判别器)
        reference_state_initialization = (
            False  # 是否从参考轨迹中初始化状态 (False则随机初始化)
        )
        standup_imitation = True  # 执行站立模仿任务 (True则从躺姿开始)

    class terrain(LeggedRobotCfg.terrain):
        """地形配置"""

        mesh_type = (
            "plane"  # 地形类型: plane(平面), trimesh(三角网格), heightfield(高度场)
        )
        # mesh_type = "trimesh"
        # mesh_type = "heightfield"
        curriculum = False  # 是否启用课程学习 (分阶段增加地形难度)
        # rough terrain only:
        measure_heights = False  # 是否测量地形高度 (用于感知地形)
        static_friction = 0.6  # 静摩擦系数
        dynamic_friction = 0.6  # 动摩擦系数
        terrain_length = 8.0  # 地形长度 [m]
        terrain_width = 8.0  # 地形宽度 [m]
        num_rows = 20  # 地形行数 (难度等级数)
        num_cols = 20  # 地形列数 (地形类型数)
        max_init_terrain_level = 10  # 课程学习的起始难度等级
        # 地形类型比例: plane(平面); obstacles(障碍物); uniform(随机); slope_up(上坡); slope_down(下坡), stair_up(上楼梯), stair_down(下楼梯)
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0  # 恢复系数 (弹性,0表示完全非弹性碰撞)

    class init_state(LeggedRobotCfg.init_state):
        """初始状态配置"""

        # 躺姿开始: 较低的高度, 关节角度接近0
        pos = [0.0, 0.0, 0.35]  # 初始位置 [x, y, z] (m)
        # 默认关节角度 (站立模仿任务通常从躺姿开始,所以关节角度设为0)
        default_joint_angles = {
            "Lleg_hip_p_joint": 0.0,  # 左髋俯仰关节
            "Lleg_hip_r_joint": 0.0,  # 左髋翻滚关节
            "Lleg_hip_y_joint": 0.0,  # 左髋偏航关节
            "Lleg_knee_joint": 0.0,  # 左膝关节
            "Lleg_ankle_p_joint": 0.0,  # 左踝俯仰关节
            "Lleg_ankle_r_joint": 0.0,  # 左踝翻滚关节
            "Rleg_hip_p_joint": 0.0,  # 右髋俯仰关节
            "Rleg_hip_r_joint": 0.0,  # 右髋翻滚关节
            "Rleg_hip_y_joint": 0.0,  # 右髋偏航关节
            "Rleg_knee_joint": 0.0,  # 右膝关节
            "Rleg_ankle_p_joint": 0.0,  # 右踝俯仰关节
            "Rleg_ankle_r_joint": 0.0,  # 右踝翻滚关节
        }
        init_joint_state_train = True  # 训练时是否随机初始化关节状态 (增加多样性)
        init_base_angle_max = 0.1  # 初始机身姿态角度扰动的最大范围 (rad, 约±5.7度)

    class control(LeggedRobotCfg.control):
        """控制器配置 (PD控制器参数)"""

        # PD控制器参数: 关节力矩 = Kp * (目标角度 - 当前角度) + Kd * (0 - 当前角速度)
        # 刚度系数 Kp [N·m/rad]
        stiffness = {
            "hip_r": 150.0,  # 髋翻滚关节
            "hip_p": 150.0,  # 髋俯仰关节
            "hip_y": 100.0,  # 髋偏航关节
            "knee": 150.0,  # 膝关节
            "ankle": 30.0,  # 踝关节 (较小,因为踝关节需要柔顺性)
        }
        # 阻尼系数 Kd [N·m·s/rad]
        damping = {
            "hip_r": 1.5,
            "hip_p": 1.5,
            "hip_y": 1.0,
            "knee": 1.5,
            "ankle": 1.0,
        }
        # 动作缩放: 目标角度 = actionScale * action + defaultAngle
        action_scale = 0.25  # 动作缩放系数 (限制动作幅度,提高稳定性)
        # 策略推理频率: 仿真步长 sim.dt = 0.005s (200Hz), decimation = 4 -> 控制频率 50Hz
        decimation = 4  # 每4个仿真步执行一次策略推理 (降低控制频率,更符合实际)

    class asset(LeggedRobotCfg.asset):
        """机器人模型资源配置"""

        file = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hhfc_sf/urdf/hhfc.urdf"  # URDF文件路径
        name = "hhfc"  # 机器人名称
        foot_name = "ankle_r"  # 脚部刚体名称 (用于接触检测)
        knee_name = "knee"  # 膝盖刚体名称
        # 哪些部位接触地面会终止 episode (空列表表示不因接触终止)
        terminate_after_contacts_on = []
        # 哪些部位接触地面会额外惩罚 (空列表表示不惩罚任何接触)
        penalize_contacts_on = []
        self_collisions = 0  # 自碰撞检测: 1禁用, 0启用 (按位过滤)
        flip_visual_attachments = False  # 是否翻转视觉附件 (某些URDF需要)
        replace_cylinder_with_capsule = False  # 是否将圆柱体替换为胶囊体 (提高碰撞性能)
        fix_base_link = False  # 是否固定机身 (True则机器人悬浮,用于调试)
        # 参考轨迹文件路径
        ref_traj = TRAJECTORY_FILE

    class rewards(LeggedRobotCfg.rewards):
        """奖励函数相关参数"""

        base_height_target = 0.92  # 目标基座高度 [m]
        foot_clearance_target = 0.08  # 摆动腿目标离地高度 [m]
        foot_height_offset = 0.068  # 脚部高度偏移基准 [m]
        only_positive_rewards = False  # 是否只使用正奖励 (False表示可以有惩罚项)
        soft_dof_pos_limit = 0.95  # 软关节位置限制 (超过此比例开始惩罚)
        soft_dof_vel_limit = 0.95  # 软关节速度限制
        about_landing_threshold = 0.08  # "即将着地"判定的高度阈值 [m]
        min_dist = 0.25  # 双脚/双膝的最小合理间距 [m]
        max_dist = 0.6  # 双脚的最大合理间距 [m]

        class scales(LeggedRobotCfg.rewards.scales):
            """各奖励项的权重系数"""

            # ========== 限制类惩罚 ==========
            termination = -200.0  # 终止惩罚 (摔倒或超时)
            dof_pos_limits = -2.0  # 关节位置超限惩罚
            dof_vel_limits = -1.0  # 关节速度超限惩罚
            # ========== 平滑性惩罚 ==========
            dof_acc = -2.5e-7  # 关节加速度惩罚 (限制关节加速度)
            dof_vel = -5.0e-4  # 关节速度惩罚 (鼓励缓慢运动)
            action_rate = -0.01  # 动作变化率惩罚 (一阶差分)
            action_smoothness = -0.01  # 动作平滑度惩罚 (二阶差分)
            torques = -1.0e-5  # 力矩惩罚 (减少能耗)
            # ========== 正则化惩罚 ==========
            collision = -1.0  # 碰撞惩罚 (不期望的身体部位接触地面)
            lin_vel_z = -0.5  # 垂直线速度惩罚 (鼓励保持高度稳定)
            ang_vel_xy = -0.05  # roll和pitch角速度惩罚 (鼓励姿态稳定)
            orientation = -5.0  # 姿态偏离惩罚 (鼓励保持直立)
            foot_landing_vel = -0.2  # 脚部着地速度惩罚 (鼓励轻柔着地)
            feet_distance = 0.2  # 双脚间距奖励 (鼓励合理步宽)
            knee_distance = 0.2  # 双膝间距奖励
            # ========== 任务奖励: 禁用运动跟踪,启用模仿学习 ==========
            tracking_lin_vel = 1e-8  # 线速度跟踪权重 (几乎为0,因为是站立任务)
            tracking_ang_vel = 1e-8  # 角速度跟踪权重 (几乎为0)
            imitation_state = 3.0  # 状态模仿奖励 exp(-state_error)
            stand_success = 2.0  # 成功站立的额外奖励 (高度和姿态满足条件时)
            # foot_clearance = 0.0  # 脚部离地高度奖励 (已注释)

    class commands(LeggedRobotCfg.commands):
        """速度命令配置"""

        num_commands = 4  # 命令维度: lin_vel_x(前进), lin_vel_y(侧移), ang_vel_yaw(转向), heading(目标朝向)
        resampling_time = 5.0  # 命令保持时间 [s] (每5秒重新采样一次命令)
        # False → 命令[2]是角速度 ang_vel_yaw
        # True → 命令[3]是目标朝向角 heading, 由环境自动换算成 ang_vel_yaw
        heading_command = False
        curriculum = True  # 是否启用课程学习 (逐渐增加命令难度)
        max_curriculum = 2.0  # 课程学习最大等级
        min_curriculum = -0.5  # 课程学习最小等级

        class ranges:
            """命令范围 [最小值, 最大值]"""

            lin_vel_x = [0.0, 0.5]  # 前进速度范围 [m/s]
            lin_vel_y = [0.0, 0.0]  # 侧移速度范围 [m/s] (0表示不侧移)
            ang_vel_yaw = [-0.5, 0.5]  # 转向角速度范围 [rad/s]
            heading = [-3.14, 3.14]  # 目标朝向范围 [rad] (当heading_command=True时使用)

    class domain_rand(LeggedRobotCfg.domain_rand):
        """域随机化配置 (提高策略泛化能力和sim2real迁移性)"""

        # ========== 摩擦系数随机化 ==========
        randomize_friction = True  # 是否随机化地面摩擦系数
        friction_range = [0.2, 1.2]  # 摩擦系数范围 (模拟不同地面材质)
        # ========== 基座质量随机化 ==========
        randomize_base_mass = True  # 是否随机化机器人质量
        added_mass_range = [-1.0, 3.0]  # 附加质量范围 [kg] (模拟负载变化)
        # ========== 质心位置随机化 ==========
        randomize_com_pos = True  # 是否随机化质心位置
        com_displacement_range = [
            [-0.05, 0.05],
            [-0.05, 0.05],
            [-0.05, 0.05],
        ]  # xyz方向的质心偏移 [m]
        # ========== 主动扰动 ==========
        push_robots = True  # 是否启用随机外力推动 (模拟外部干扰)
        push_interval_s = 10  # 推动间隔 [s]
        max_push_vel_xy = 1.0  # 最大推动速度 [m/s] (通过直接设置基座速度实现)
        # ========== 控制延迟随机化 ==========
        randomize_ctrl_delay = False  # 是否随机化控制延迟 (模拟通信延迟)
        ctrl_delay_step_range = [0, 2]  # 延迟步数范围 [步] (0-2步 = 0-10ms延迟 @200Hz)
        # ========== PD增益随机化 ==========
        randomize_pd_gain = True  # 是否随机化PD控制器增益
        kp_range = [0.8, 1.2]  # Kp缩放范围 (实际Kp = 配置Kp × 此缩放)
        kd_range = [0.8, 1.2]  # Kd缩放范围
        # ========== 关节转动惯量随机化 ==========
        randomize_joint_armature = True  # 是否随机化关节附加转动惯量
        joint_armature_range = [
            0.0,
            0.1,
        ]  # 转动惯量范围 [N·m·s/rad] (模拟关节摩擦和惯性)

    class normalization(LeggedRobotCfg.normalization):
        """归一化配置 (用于网络输入/输出)"""

        clip_observations = 100.0  # 观测值裁剪阈值 (防止异常值)
        clip_actions = 10.0  # 动作裁剪阈值 (防止异常大的动作)

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            """观测值缩放系数 (将物理量缩放到合适的数值范围)"""

            lin_vel = 2.0  # 线速度缩放 (除以2,将±1m/s映射到±0.5)
            ang_vel = 0.25  # 角速度缩放 (除以4,将±4rad/s映射到±1)
            dof_pos = 1.0  # 关节位置缩放 (保持原值)
            dof_vel = 0.05  # 关节速度缩放 (除以20,将±20rad/s映射到±1)
            height_measurements = 5.0  # 高度测量缩放

    class noise(LeggedRobotCfg.noise):
        """观测噪声配置 (模拟传感器误差)"""

        add_noise = True  # 是否添加观测噪声
        noise_level = 1.0  # 全局噪声缩放系数 (1.0表示使用下面定义的标准噪声水平)

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            """各观测量的噪声标准差 (均匀分布 U(-scale, scale))"""

            dof_pos = 0.05  # 关节位置噪声 [rad]
            dof_vel = 0.5  # 关节速度噪声 [rad/s]
            lin_vel = 0.05  # 线速度噪声 [m/s]
            ang_vel = 0.1  # 角速度噪声 [rad/s]
            gravity = 0.05  # 重力方向噪声
            height_measurements = 0.1  # 高度测量噪声 [m]

    class sim(LeggedRobotCfg.sim):
        """物理仿真参数配置"""

        dt = 0.005  # 仿真步长 [s] -> 200Hz (每秒仿真200步)
        substeps = 1  # 每个仿真步长内的PhysX子步数 (1表示不细分)
        up_axis = 1  # 世界坐标系的"竖直向上"方向: 0是y轴, 1是z轴 (Isaac Gym使用z轴向上)

        class physx(LeggedRobotCfg.sim.physx):
            """PhysX物理引擎参数"""

            num_threads = 10  # 物理引擎CPU线程数 (多线程加速)
            solver_type = 1  # 物理解算器类型: 0是PGS(投影高斯-赛德尔), 1是TGS(截断高斯-赛德尔,更稳定)
            num_position_iterations = 4  # 位置约束迭代次数 (越多越精确但越慢)
            num_velocity_iterations = 1  # 速度约束迭代次数
            contact_offset = 0.01  # 接触判定阈值 [m] (物体距离<此值时认为接触)
            rest_offset = 0.0  # 接触保持间隙 [m] (接触时的目标分离距离)
            bounce_threshold_velocity = (
                0.1  # 弹性碰撞阈值速度 [m/s] (低于此速度的碰撞视为非弹性)
            )
            max_depenetration_velocity = (
                1.0  # 最大渗透修正速度 [m/s] (防止物体穿透时修正过快)
            )
            max_gpu_contact_pairs = 2**24  # GPU上能同时处理的最大接触对数 (约1670万)
            default_buffer_size_multiplier = (
                5  # GPU内存缓冲区大小倍增因子 (增大可容纳更多物体/接触)
            )
            contact_collection = 2  # 接触点收集策略: 0不收集, 1只收集最后一个子步, 2收集所有子步 (2最精确)


class HhfcRobotCfgGAIL(LeggedRobotCfgGAIL):
    """Hhfc机器人的GAIL/AMP训练配置类

    继承自LeggedRobotCfgGAIL,定义了强化学习训练相关的所有超参数。
    """

    class runner(LeggedRobotCfgGAIL.runner):
        """训练运行器配置"""

        experiment_name = "hhfc_amp"  # 实验名称 (用于日志和模型保存目录)
        run_name = "24"  # 运行名称/标签
        num_steps_per_env = 24  # 每个环境每次rollout的步数 (影响策略更新频率)
        max_iterations = 20000  # 策略迭代最大次数 (训练总步数 = max_iterations × num_envs × num_steps_per_env)
        num_state = NUM_STATES  # 状态维度 (用于判别器输入)
        ref_traj = TRAJECTORY_FILE  # 参考轨迹文件路径 (用于AMP/GAIL算法)
        save_interval = 200  # 模型保存间隔 (每200次迭代保存一次)
        resume = False  # 是否断点续训 (True则从load_run加载模型继续训练)
        load_run = (
            -1
        )  # 加载哪个运行的模型: -1自动加载最新模型, 或指定运行名如'Jul15_15-50-28_24'

    class algorithm(LeggedRobotCfgGAIL.algorithm):
        """PPO算法超参数"""

        entropy_coef = 0.01  # 熵正则化系数 (鼓励探索,防止策略过早收敛)
        num_mini_batches = 1  # 每次rollout的数据分成几份做梯度下降 (1表示不分批)
        learning_rate = 2.5e-4  # Adam优化器学习率

    class policy:
        """策略网络结构配置"""

        init_noise_std = 1.0  # 策略网络输出动作的初始噪声标准差 (影响初始探索程度)
        actor_hidden_dims = [512, 256, 128]  # Actor网络隐藏层维度 (3层MLP)
        critic_hidden_dims = [512, 256, 128]  # Critic网络隐藏层维度
        activation = "elu"  # 激活函数: elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class discriminator:
        """判别器网络结构配置 (用于AMP/GAIL算法)"""

        init_noise_std = 1.0  # 判别器输出的初始噪声标准差
        hidden_dims = [1024, 512]  # 判别器隐藏层维度 (2层MLP)
