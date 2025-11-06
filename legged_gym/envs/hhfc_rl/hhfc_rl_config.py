from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class HhfcRlRobotCfg(LeggedRobotCfg):
    """Hhfc机器人的纯强化学习环境配置类

    继承自LeggedRobotCfg基类,定义了Hhfc人形机器人通过纯强化学习站立的所有配置参数。
    与hhfc任务的区别: 不使用模仿学习,完全依靠奖励函数引导站立动作。
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
        is_amp = False  # 不使用 AMP (纯强化学习不需要判别器)
        reference_state_initialization = False  # 随机初始化 (不从参考轨迹初始化)
        standup_imitation = False  # 不执行模仿任务 (纯RL自主学习站立)

    class terrain(LeggedRobotCfg.terrain):
        """地形配置"""

        mesh_type = (
            "plane"  # 地形类型: plane(平面), trimesh(三角网格), heightfield(高度场)
        )
        curriculum = False  # 是否启用课程学习 (分阶段增加地形难度)
        measure_heights = False  # 是否测量地形高度 (用于感知地形)
        static_friction = 0.6  # 静摩擦系数
        dynamic_friction = 0.6  # 动摩擦系数
        terrain_length = 8.0  # 地形长度 [m]
        terrain_width = 8.0  # 地形宽度 [m]
        num_rows = 20  # 地形行数 (难度等级数)
        num_cols = 20  # 地形列数 (地形类型数)
        max_init_terrain_level = 10  # 课程学习的起始难度等级
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0  # 恢复系数 (弹性,0表示完全非弹性碰撞)

    class init_state(LeggedRobotCfg.init_state):
        """初始状态配置"""

        # 躺姿开始: 较低的高度, 关节角度接近0
        pos = [0.0, 0.0, 0.35]  # 初始位置 [x, y, z] (m)
        # 默认关节角度 (从躺姿开始学习站立)
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
            """各奖励项的权重系数 (针对纯RL优化)"""

            # ========== 核心任务奖励 (纯RL版本大幅增强) ==========
            stand_success = (
                10.0  # 成功站立的奖励 (主要驱动力! 高度>0.85m且姿态稳定时触发)
            )
            base_height = 3.0  # 基座高度奖励 (鼓励机器人抬高身体)

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

            # ========== 禁用运动跟踪和模仿学习 ==========
            tracking_lin_vel = 0.0  # 禁用线速度跟踪 (站立任务不需要移动)
            tracking_ang_vel = 0.0  # 禁用角速度跟踪

    class commands(LeggedRobotCfg.commands):
        """速度命令配置"""

        num_commands = 4  # 命令维度: lin_vel_x(前进), lin_vel_y(侧移), ang_vel_yaw(转向), heading(目标朝向)
        resampling_time = 5.0  # 命令保持时间 [s] (每5秒重新采样一次命令)
        heading_command = False  # False → 使用角速度命令

        class ranges(LeggedRobotCfg.commands.ranges):
            """命令采样范围 (站立任务命令接近0)"""

            lin_vel_x = [0.0, 0.0]  # 前向速度范围 [m/s] (站立任务不需要移动)
            lin_vel_y = [0.0, 0.0]  # 侧向速度范围 [m/s]
            ang_vel_yaw = [0.0, 0.0]  # 偏航角速度范围 [rad/s]
            heading = [0.0, 0.0]  # 目标朝向范围 [rad]

    class domain_rand(LeggedRobotCfg.domain_rand):
        """域随机化配置 (提高sim2real鲁棒性)"""

        randomize_friction = True  # 随机化摩擦系数
        friction_range = [0.2, 1.2]  # 摩擦系数随机范围
        randomize_base_mass = True  # 随机化基座质量
        added_mass_range = [-1.0, 3.0]  # 质量增量范围 [kg]
        randomize_base_com = True  # 随机化质心位置
        added_com_range_x = [-0.05, 0.05]  # 质心x方向偏移 [m]
        added_com_range_y = [-0.05, 0.05]  # 质心y方向偏移 [m]
        added_com_range_z = [-0.05, 0.05]  # 质心z方向偏移 [m]
        push_robots = True  # 是否随机推动机器人 (测试稳定性)
        push_interval_s = 10  # 推动间隔 [s]
        max_push_vel_xy = 1.0  # 最大推动速度 [m/s]
        randomize_ctrl_delay = True  # 随机化控制延迟
        ctrl_delay_step_range = [0, 3]  # 控制延迟步数范围 [steps]
        randomize_gains = True  # 随机化PD增益
        stiffness_multiplier_range = [0.8, 1.2]  # 刚度倍增系数范围
        damping_multiplier_range = [0.8, 1.2]  # 阻尼倍增系数范围
        randomize_joint_armature = True  # 随机化关节转动惯量
        joint_armature_multiplier_range = [0.8, 1.2]  # 转动惯量倍增系数范围

    class normalization(LeggedRobotCfg.normalization):
        """观测归一化配置"""

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


class HhfcRlRobotCfgPPO(LeggedRobotCfgPPO):
    """Hhfc机器人的纯PPO训练配置类

    继承自LeggedRobotCfgPPO,定义了纯强化学习训练相关的所有超参数。
    """

    class runner(LeggedRobotCfgPPO.runner):
        """训练运行器配置"""

        experiment_name = "hhfc_rl"  # 实验名称 (用于日志和模型保存目录)
        run_name = "ppo"  # 运行名称/标签
        num_steps_per_env = 24  # 每个环境每次rollout的步数 (影响策略更新频率)
        max_iterations = 20000  # 策略迭代最大次数 (训练总步数 = max_iterations × num_envs × num_steps_per_env)
        save_interval = 200  # 模型保存间隔 (每200次迭代保存一次)
        resume = False  # 是否断点续训 (True则从load_run加载模型继续训练)
        load_run = -1  # 加载哪个运行的模型: -1自动加载最新模型, 或指定运行名
        checkpoint = -1  # 加载哪个checkpoint: -1自动加载最新, 或指定迭代数

    class algorithm(LeggedRobotCfgPPO.algorithm):
        """PPO算法超参数"""

        entropy_coef = 0.01  # 熵正则化系数 (鼓励探索,防止策略过早收敛)
        num_learning_epochs = 5  # 每次rollout数据重复使用次数
        num_mini_batches = 4  # 每次rollout的数据分成几份做梯度下降
        learning_rate = 5.0e-4  # Adam优化器学习率 (纯RL可以稍高一些)
        schedule = "adaptive"  # 学习率调度: adaptive(自适应), fixed(固定)
        gamma = 0.99  # 折扣因子 (未来奖励的衰减系数)
        lam = 0.95  # GAE lambda (优势函数估计的平滑参数)
        desired_kl = 0.01  # 目标KL散度 (用于自适应学习率)
        max_grad_norm = 1.0  # 梯度裁剪阈值 (防止梯度爆炸)

    class policy:
        """策略网络结构配置"""

        init_noise_std = 1.0  # 策略网络输出动作的初始噪声标准差 (影响初始探索程度)
        actor_hidden_dims = [512, 256, 128]  # Actor网络隐藏层维度 (3层MLP)
        critic_hidden_dims = [512, 256, 128]  # Critic网络隐藏层维度
        activation = "elu"  # 激活函数: elu, relu, selu, crelu, lrelu, tanh, sigmoid
