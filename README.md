# Legged Gym

*Read this in other languages: [English](README.md), [中文](README.zh-CN.md)*

## Installation
1. Create a new python virtual env with python 3.8
    - `conda create --name legged_gym python=3.8 -y`
    - `conda activate legged_gym`
2. Install pytorch with cuda-11.8:
    - `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`
4. Install this repo
   - `cd legged_gym-hhfc && pip install -e .`
   - `cd rsl_rl && pip install -e .` (install rsl_rl)
   - `cd .. && pip install -e .` (install legged_gym)
   - `pip install tensorboard pandas` (addtional modules)

## Code Structure
1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one containing all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `legged_gym/envs/__init__.py`, but can also be done from outside of this repository.  

## Usage
1. **Train**:  
   ```bash
   cd legged_gym-hhfc/legged_gym/scripts && python train.py
   ```
   - To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
   - To run headless (no rendering) add `--headless`.
   - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
   - The trained policy is saved in `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
   - The following command line arguments override the values set in the config files:
     - `--task TASK`: Task name.
     - `--resume`: Resume training from a checkpoint
     - `--experiment_name EXPERIMENT_NAME`: Name of the experiment to run or load.
     - `--run_name RUN_NAME`: Name of the run.
     - `--load_run LOAD_RUN`: Name of the run to load when resume=True. If -1: will load the last run.
     - `--checkpoint CHECKPOINT`: Saved model checkpoint number. If -1: will load the last checkpoint.
     - `--num_envs NUM_ENVS`: Number of environments to create.
     - `--seed SEED`: Random seed.
     - `--max_iterations MAX_ITERATIONS`: Maximum number of training iterations.

2. **Play a trained policy**:  
   ```bash
   python issacgym_anymal/scripts/play.py --task=anymal_c_flat
   ```
   - By default the loaded policy is the last model of the last run of the experiment folder.
   - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

3. **Visualize reference data**:
   ```bash
   cd legged_gym-hhfc/legged_gym/scripts && python test_humanoid_ref_data.py
   ```

## Adding a New Environment
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and no reward scales. 

1. Add a new folder to `envs/` with `<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resources/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in `<your_env>.py`, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `isaacgym_anymal/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!
