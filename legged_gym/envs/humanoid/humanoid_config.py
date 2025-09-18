from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgGAIL

# 参考轨迹文件
TRAJECTORY_FILE = f"{LEGGED_GYM_ROOT_DIR}/resources/trajectory/humanoid/standup01.dat"
# 状态维度
NUM_STATES = 30


class HumanoidRobotCfg(LeggedRobotCfg):
    seed = 42

    class env(LeggedRobotCfg.env):
        frame_stack = 10  # actor 网络保留 10 帧上下文
        c_frame_stack = 3  # critic 网络保留 3 帧上下文
        # commands(3)+dof_pos(12)+dof_vel(12)+
        # actions(12)+base_ang_vel(3)+gravity(3)=45
        num_single_obs = 45
        num_observations = int(frame_stack * num_single_obs)  # 实际输入到网络的观测维度
        # 单帧的特权观测，即训练时 critic 网络才能使用的额外信息
        # privileged obs + base height -> 81
        single_num_privileged_obs = 81
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12  # 动作维度
        num_envs = 8192  # 并行环境数量
        episode_length_s = 20  # 每个 episode 最长 20 秒（仿真时间）
        is_amp = True  # 使用 AMP 算法
        num_states = NUM_STATES  # 状态维度
        reference_state_initialization = False  # 是否从参考轨迹中初始化状态
        standup_imitation = True  # 执行站立模仿任务

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        # mesh_type = "trimesh"
        # mesh_type = "heightfield"
        curriculum = False  # 分阶段学习
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0

    class init_state(LeggedRobotCfg.init_state):
        # lying start: low height, flat joints
        pos = [0.0, 0.0, 0.35]
        default_joint_angles = {
            "Lleg_hip_p_joint": 0.0,
            "Lleg_hip_r_joint": 0.0,
            "Lleg_hip_y_joint": 0.0,
            "Lleg_knee_joint": 0.0,
            "Lleg_ankle_p_joint": 0.0,
            "Lleg_ankle_r_joint": 0.0,
            "Rleg_hip_p_joint": 0.0,
            "Rleg_hip_r_joint": 0.0,
            "Rleg_hip_y_joint": 0.0,
            "Rleg_knee_joint": 0.0,
            "Rleg_ankle_p_joint": 0.0,
            "Rleg_ankle_r_joint": 0.0,
        }
        init_joint_state_train = True  # 随机初始化关节状态
        init_base_angle_max = 0.1  # 初始机身姿态角度扰动的最大范围（rad）

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # 刚度系数
        stiffness = {
            "hip_r": 150.0,
            "hip_p": 150.0,
            "hip_y": 100.0,
            "knee": 150.0,
            "ankle": 30.0,
        }
        # 阻尼系数
        damping = {
            "hip_r": 1.5,
            "hip_p": 1.5,
            "hip_y": 1.0,
            "knee": 1.5,
            "ankle": 1.0,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  # 动作缩放系数（作用于 tanh 之前的动作值）
        # 策略推理频率: 仿真步长 sim.dt = 0.005   decimation = 4 -> 50Hz
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/hhfc_sf/urdf/hhfc.urdf"
        name = "hhfc"
        foot_name = "ankle_r"
        knee_name = "knee"
        # 哪些部位接触地面会终止 episode
        terminate_after_contacts_on = []
        # 哪些部位接触地面会额外惩罚
        penalize_contacts_on = []
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False  # 是否翻转视觉附件
        replace_cylinder_with_capsule = False  # 是否更换模型中的圆柱体为胶囊体
        fix_base_link = False  # 是否固定机身
        # 参考轨迹文件
        ref_traj = TRAJECTORY_FILE

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.92
        foot_clearance_target = 0.08
        foot_height_offset = 0.068
        only_positive_rewards = False
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.95
        about_landing_threshold = 0.08
        min_dist = 0.25
        max_dist = 0.6

        class scales(LeggedRobotCfg.rewards.scales):
            # limit
            termination = -200.0
            dof_pos_limits = -2.0
            dof_vel_limits = -1.0
            # smooth
            dof_acc = -2.5e-7
            dof_vel = -5.0e-4
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -1.0e-5
            # regularization
            collision = -1.0
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -5.0
            foot_landing_vel = -0.2
            feet_distance = 0.2
            knee_distance = 0.2
            # task: disable locomotion tracking & add imitation
            tracking_lin_vel = 1e-8
            tracking_ang_vel = 1e-8
            imitation_state = 3.0  # exp(-state error)
            stand_success = 2.0  # bonus when upright
            # foot_clearance = 0.0

    class commands(LeggedRobotCfg.commands):
        num_commands = 4  # 指令: lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 5.0  # 命令保持时间[s]
        # False → 指令是角速度 ang_vel_yaw
        # True → 指令是目标朝向角 heading，由环境换算成 ang_vel_yaw
        heading_command = False
        curriculum = True
        max_curriculum = 2.0
        min_curriculum = -0.5

        class ranges:
            lin_vel_x = [0.0, 0.5]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            heading = [-3.14, 3.14]  # min max [rad]

    class domain_rand(LeggedRobotCfg.domain_rand):
        # 摩擦系数
        randomize_friction = True
        friction_range = [0.2, 1.2]
        # 基座质量
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        # 质心位置
        randomize_com_pos = True
        com_displacement_range = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]
        # 主动扰动
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0
        # 控制延迟
        randomize_ctrl_delay = False
        ctrl_delay_step_range = [0, 2]  # 0-20ms delay
        # PD增益
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        # 关节附加转动惯量
        randomize_joint_armature = True
        joint_armature_range = [0.0, 0.1]  # [Nms/rad]

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 100.0  # 观测值裁剪阈值
        clip_actions = 10.0  # 动作裁剪阈值

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # 全局噪声缩放

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.05
            dof_vel = 0.5
            lin_vel = 0.05
            ang_vel = 0.1
            gravity = 0.05
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 仿真步长 -> 200Hz
        substeps = 1  # 每个仿真步长内的子步数
        up_axis = 1  # 世界坐标系的“竖直方向”: 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10  # 物理引擎线程数
            solver_type = 1  # 物理解算器类型: 0 is pgs, 1 is tgs
            num_position_iterations = 4  # 位置约束迭代次数
            num_velocity_iterations = 1  # 速度约束迭代次数
            contact_offset = 0.01  # 接触判定阈值 [m]
            rest_offset = 0.0  # 接触保持间隙 [m]
            bounce_threshold_velocity = 0.1  # 弹性碰撞阈值 [m/s]
            max_depenetration_velocity = 1.0  # 最大渗透修正速度 [m/s]
            max_gpu_contact_pairs = 2**24  # GPU 上能同时处理的最大接触对数
            default_buffer_size_multiplier = 5  # GPU 内存缓冲区大小倍增因子
            contact_collection = (
                2  # 接触点收集: 0 is 不收集, 1 is 最后一个子步, 2 is 所有子步
            )


class HumanoidRobotCfgGAIL(LeggedRobotCfgGAIL):

    class runner(LeggedRobotCfgGAIL.runner):
        experiment_name = "humanoid_amp"  # 实验名称
        run_name = "24"  # 运行名称
        num_steps_per_env = 24  # 策略更新频率: 每个环境每次 rollout 的步数
        max_iterations = 20000  # 策略迭代最大次数
        num_state = NUM_STATES  # 状态维度
        ref_traj = TRAJECTORY_FILE  # 参考轨迹文件
        save_interval = 200  # 模型保存间隔
        resume = False  # 断点续训
        load_run = -1  # -1 自动加载最新模型 或 指定 'Jul15_15-50-28_24'

    class algorithm(LeggedRobotCfgGAIL.algorithm):
        entropy_coef = 0.01  # 熵正则化系数
        num_mini_batches = 1  # 每次 rollout 的数据分成几份做梯度下降
        learning_rate = 2.5e-4  # 学习率

    class policy:
        init_noise_std = 1.0  # 策略网络输出动作的初始噪声标准差
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class discriminator:
        init_noise_std = 1.0  # 判别器网络输出动作的初始噪声标准差
        hidden_dims = [1024, 512]
