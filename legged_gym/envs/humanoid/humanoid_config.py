from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, LeggedRobotCfgGAIL
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch

# ! Specify trajectory file here
TRAJECTORY_FILE = f"{LEGGED_GYM_ROOT_DIR}/resources/trajectory/humanoid/run_50Hz_4p0_24.dat"
NUM_STATES = 24

class HumanoidRobotCfg(LeggedRobotCfg):
    seed = 0
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 10   # Policy frame stack number
        c_frame_stack = 3  # Critic frame stack number
        num_single_obs = 45
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 63 + 17
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 8192
        episode_length_s = 20  # episode length in seconds
        
        is_amp = True
        num_states = NUM_STATES  # discriminator input
        reference_state_initialization = True  # initialize state from reference data
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        # mesh_type = "trimesh"
        # mesh_type = "heightfield"
        curriculum = False
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
        pos = [0.0, 0.0, 0.98] # [0.0, 0.0, 0.95]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "Lleg_hip_p_joint": 0.3,
            "Lleg_hip_r_joint": 0.0,
            "Lleg_hip_y_joint": 0.0,
            "Lleg_knee_joint": -0.6,
            "Lleg_ankle_p_joint": -0.3,
            "Lleg_ankle_r_joint": 0.0,
            "Rleg_hip_p_joint": 0.3,
            "Rleg_hip_r_joint": 0.0,
            "Rleg_hip_y_joint": 0.0, 
            "Rleg_knee_joint": -0.6,
            "Rleg_ankle_p_joint": -0.3,
            "Rleg_ankle_r_joint": 0.0,
            # "Larm_shoulder_p_joint": 0.0,
            # "Rarm_shoulder_p_joint": 0.0,
        }
        init_joint_state_train = True
        init_base_angle_max = 0.1 # rad

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        ## 1
        stiffness = {
            "hip_r": 150.0,
            "hip_p": 150.0,
            "hip_y": 100.0,
            "knee": 150.0,
            "ankle": 30,
        }
        damping = {
            "hip_r": 1.5,
            "hip_p": 1.5,
            "hip_y": 1.0,
            "knee": 1.5,
            "ankle": 1.0,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4 # 50Hz

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/hhfc_sf/urdf/hhfc.urdf"
        name = "hhfc"
        foot_name = "ankle_r"
        knee_name = "knee"
        terminate_after_contacts_on = ["base"]
        penalize_contacts_on = ["knee", "hip"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
        
        # reference trajectory file (for amp)
        ref_traj = TRAJECTORY_FILE

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.92
        foot_clearance_target = 0.08
        foot_height_offset = 0.068 # offset of the foot height due to coordinate definition
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
            dof_vel = -5.e-4
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -1.e-5
            # regularization
            collision = -1.0
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -5.0
            foot_landing_vel = -0.2
            feet_distance = 0.2
            knee_distance = 0.2
            
            # task
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # foot_clearance = 0.0
    
    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 5.0    # time before command are changed[s]
        heading_command = False  # if true, the command is a heading angle instead of an angular velocity
        curriculum = True
        max_curriculum = 2.0
        min_curriculum = -0.5
        
        class ranges:
            lin_vel_x = [0.0, 0.5]    # min max [m/s]
            lin_vel_y = [0.0, 0.0]    # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            heading = [-3.14, 3.14]
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.2]

        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]

        randomize_com_pos = True
        com_displacement_range = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]

        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0
        
        randomize_ctrl_delay = False
        ctrl_delay_step_range = [0, 2] # 0-20ms delay
        
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        
        randomize_joint_armature = True
        joint_armature_range = [0.0, 0.1]  # [Nms/rad]
    
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 10.
    
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.05
            dof_vel = 0.5
            lin_vel = 0.05
            ang_vel = 0.1
            gravity = 0.05
            height_measurements = 0.1
    
    class sim(LeggedRobotCfg.sim):
        dt =  0.005
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class HumanoidRobotCfgGAIL(LeggedRobotCfgGAIL):
    class runner(LeggedRobotCfgGAIL.runner):
        experiment_name = 'humanoid_amp'
        num_steps_per_env = 24
        max_iterations = 6000   # number of policy updates
        num_state = NUM_STATES  # 27 for novel, 30 for hop
        ref_traj = TRAJECTORY_FILE

        # logging
        save_interval = 200  # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = "Jul15_15-50-28_24"  # -1 = last run
        run_name = '24'

    class algorithm(LeggedRobotCfgGAIL.algorithm):
        entropy_coef = 0.01
        num_mini_batches = 1  # 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2.5e-4  # 1.e-3 #5.e-4

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class discriminator:
        init_noise_std = 1.0
        hidden_dims = [1024, 512]

