import numpy as np
import scipy.io as spio
from legged_gym import LEGGED_GYM_ROOT_DIR
# from legged_gym.envs import LeggedRobot

# relative paths to the two available trajectories
PATH_CONSTANT_SPEED = LEGGED_GYM_ROOT_DIR + '/resources/trajectory/drloco_mat/Trajecs_Constant_Speed_400Hz.mat'
PATH_SPEED_RAMP = LEGGED_GYM_ROOT_DIR + '/resources/trajectory/drloco_mat/Trajecs_Ramp_Slow_400Hz_EulerTrunkAdded.mat'
# path to the reference trajectories to use
PATH_REF_TRAJECS = PATH_SPEED_RAMP

# is the trajectory with the constant speed chosen?
_is_constant_speed = PATH_CONSTANT_SPEED in PATH_REF_TRAJECS

# label every trajectory in the mocap data with the corresponding name
labels = ['COM Pos (X)', 'COM Pos (Y)', 'COM Pos (Z)',
          'Trunk Rot (quat,w)', 'Trunk Rot (quat,x)', 'Trunk Rot (quat,y)', 'Trunk Rot (quat,z)',
          'Ang Hip Frontal R', 'Ang Hip Sagittal R',
          'Ang Knee R', 'Ang Ankle R',
          'Ang Hip Frontal L', 'Ang Hip Sagittal L',
          'Ang Knee L', 'Ang Ankle L',

          'COM Vel (X)', 'COM Vel (Y)', 'COM Vel (Z)',
          'Trunk Ang Vel (X)', 'Trunk Ang Vel (Y)', 'Trunk Ang Vel (Z)',
          'Vel Hip Frontal R', 'Vel Hip Sagittal R',
          'Vel Knee R', 'Vel Ankle R',
          'Vel Hip Frontal L', 'Vel Hip Sagittal L',
          'Vel Knee L', 'Vel Ankle L',

          'Foot Pos L (X)', 'Foot Pos L (Y)', 'Foot Pos L (Z)',
          'Foot Pos R (X)', 'Foot Pos R (Y)', 'Foot Pos R (Z)',

          'GRF R', 'GRF L',

          'Trunk Rot (euler,x)', 'Trunk Rot (euler,y)', 'Trunk Rot (euler,z)',
          ]

if _is_constant_speed:
    labels.remove('GRF R')
    labels.remove('GRF L')

labels = np.array(labels)

# reference trajectory: joint position indices
COM_POSX, COM_POSY, COM_POSZ = range(0,3)
TRUNK_ROT_Q1, TRUNK_ROT_Q2, TRUNK_ROT_Q3, TRUNK_ROT_Q4 = range(3,7)
HIP_FRONT_ANG_R, HIP_SAG_ANG_R, KNEE_ANG_R, ANKLE_ANG_R = range(7,11)
HIP_FRONT_ANG_L, HIP_SAG_ANG_L, KNEE_ANG_L, ANKLE_ANG_L = range(11,15)

# reference trajectory: joint velocity indices
COM_VELX, COM_VELY, COM_VELZ = range(15,18)
TRUNK_ANGVEL_X, TRUNK_ANGVEL_Y, TRUNK_ANGVEL_Z = range(18,21)
HIP_FRONT_ANGVEL_R, HIP_SAG_ANGVEL_R, KNEE_ANGVEL_R, ANKLE_ANGVEL_R = range(21,25)
HIP_FRONT_ANGVEL_L, HIP_SAG_ANGVEL_L, KNEE_ANGVEL_L, ANKLE_ANGVEL_L = range(25,29)

# reference trajectory: foot position and GRF indices
FOOT_POSX_L, FOOT_POSY_L, FOOT_POSZ_L, FOOT_POSX_R, FOOT_POSY_R, FOOT_POSZ_R = range(29,35)

if _is_constant_speed:
    TRUNK_ROT_X, TRUNK_ROT_Y, TRUNK_ROT_Z = range(35, 38)
else:
    GRF_R, GRF_L = range(35, 37)
    TRUNK_ROT_X, TRUNK_ROT_Y, TRUNK_ROT_Z = range(37, 40)


class DrlocoReferenceTrajectories:
    def __init__(self, num_env, decimation, cfg_indices):
        self.num_env = num_env
        self._qpos_indices = cfg_indices["qpos_indices"]
        self._qvel_indices = cfg_indices["qvel_indices"]
        self._base_indices = cfg_indices["base_indices"]
        self._state_indices = cfg_indices["state_indices"]
        self._decimation = decimation

        # containers of the joint positions and velocities of the whole reference trajectory and record the [cell, traj_len]
        self._load_ref_trajecs()
        self.step = np.zeros((self.num_env,1), dtype=int)
        self.phi = np.zeros((self.num_env,1), dtype=float)
        self.state_full = np.zeros((self.num_env, 26))
        self.state_full_3d = np.zeros((self.num_env, 30))
        self.env_traj_len = np.zeros((self.num_env, 2), dtype=int)

        self.is_3dhip = True


    def _load_ref_trajecs(self):
        """ In this class, the data is split into individual steps.
            Shape of data is: (n_steps, data_dims, traj_len). """
        # load matlab data, containing trajectories of 250 steps
        path = PATH_REF_TRAJECS
        data = spio.loadmat(path, squeeze_me=True)
        # 250 steps, shape (250,1) for ramp data, where 1 is an array with kinematic data
        data = data['Data']
        # flatten the array to have dim (steps,)
        self.data = data.flatten()
        self.cell_len = self.data.shape[0]
        traj_len = []
        for i in range(0,self.cell_len):
            traj_len.append(self.data[i].shape[1] - 1)
        self.cell_max_traj_len = np.array(traj_len)
        self.num_joint = self.data[0].shape[0]

    def next(self):
        """
        Increase the internally managed position on the reference trajectory.
        This method should be called in the step() function of the RL environment.
        """
        self.step += self._decimation
        # reset the position to zero again when it reached the max possible value
        for i in range(len(self.step)):
            # i represents the env_id, [cell_step, traj_len_max]
            if self.step[i] >= self.env_traj_len[i, 1]:
                self.step[i] = self.step[i] - self.env_traj_len[i, 1]
                self.env_traj_len[i, 0] = self.env_traj_len[i, 0] + 1
                if self.env_traj_len[i, 0] >= 250:
                    self.env_traj_len[i, 0] = 0
                self.env_traj_len[i, 1] = self.cell_max_traj_len[self.env_traj_len[i, 0]]

        self.phi = self.step.flatten() / self.env_traj_len[:, 1]
        self.phi = self.phi[:, np.newaxis]
        return self.phi

    def get_ref_observation(self):
        """
        Returns the reference information of the considered robot model
        at the current timestep/(time)position.
        """
        for i in range(self.num_env):
            # a = self.data[self.env_traj_len[i, 0]][self._state_indices, self.step[i]]
            self.state_full[i,:] = self.data[self.env_traj_len[i, 0]][self._state_indices, self.step[i]]

        # 3d_hip: add hip traversal value
        angle_add = 0.05 * np.ones([self.num_env,1])
        vel_add = np.zeros([self.num_env,1])
        self.state_full_3d = np.concatenate([self.state_full[:, :12], -angle_add, self.state_full[:, 12:16], angle_add,
                                             self.state_full[:, 16:20], vel_add, self.state_full[:, 20:24], vel_add, self.state_full[:, 24:26]], axis=1)

        qpos = self.state_full_3d[:, self._qpos_indices]
        qvel = self.state_full_3d[:, self._qvel_indices]
        base = self.state_full_3d[:, self._base_indices]
        return base, qpos, qvel

    def get_random_init_state(self, env_ids):
        """
        Random State Initialization (cf. DeepMimic Paper).
        :returns qpos and qvel of a random position on the reference trajectories
        """
        for i in env_ids:
            cell_step = np.random.randint(self.cell_len)
            max_traj_len = self.cell_max_traj_len[cell_step]
            self.step[i] = np.random.choice(max_traj_len)
            self.env_traj_len[i] = [cell_step, max_traj_len]
        return self.get_ref_observation()

    # def get_deterministic_init_state(self, env_ids, pos_in_percent=0):
    #     """
    #     Returns the data on a certain position on the reference trajectory.
    #     :param pos_in_percent:  specifies the position on the reference trajectory
    #                             by skipping the percentage (in [0:100]) of points
    #     """
    #     self.step = int(self._trajec_len * pos_in_percent / 100)
    #     return self.get_qpos(), self.get_qvel()

