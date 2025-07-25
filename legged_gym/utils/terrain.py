# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            # terrain = self.make_terrain(choice, difficulty)
            terrain = terrain_utils.SubTerrain("terrain",
                                               width=self.width_per_env_pixels,
                                               length=self.width_per_env_pixels,
                                               vertical_scale=self.cfg.vertical_scale,
                                               horizontal_scale=self.cfg.horizontal_scale)
            # terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.40, step_height=-0.2, platform_size=2.0)
            # terrain_utils.stairs_terrain(terrain, slope=0.4)
            # sloped_terrain_improved(terrain, slope=0.4)
            stairs_terrain_improved(terrain, step_width=0.50, step_height=0.2)
            self.add_terrain_to_map(terrain, i, j)


    # curiculum方法，地形依次增加难度
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4  # 0.4
        # step_height = 0.05 + 0.18 * difficulty
        step_height = 0.05 + 0.25 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=0.)
            # sloped_terrain_improved(terrain, slope=slope)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=0.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            # stairs_terrain_improved(terrain, step_width=0.40, step_height=step_height)
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.50, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            # terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
            terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=0.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def sloped_terrain_improved(terrain, slope=0.4):
    # x方向分为5段: 平->坡->平->坡->平
    seg1_x = np.arange(0, int(terrain.width / 16))
    seg2_x = np.arange(int(terrain.width / 16), int(terrain.width * 7 / 16))
    seg3_x = np.arange(int(terrain.width * 7 / 16), int(terrain.width * 9 / 16))
    seg4_x = np.arange(int(terrain.width * 9 / 16), int(terrain.width * 15 / 16))
    seg5_x = np.arange(int(terrain.width * 15 / 16), int(terrain.width))
    slope_width = int(terrain.width * 6 / 16) + 1
    y = np.arange(0, terrain.length)
    # 斜坡宽度为width*5/16, 由输入的坡度计算最大高度
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width * 5 / 16)
    # 生成坐标
    seg2_x_reverse = seg2_x[::-1]
    seg2_xx, yy1 = np.meshgrid(seg2_x_reverse, y, sparse=True)
    seg4_xx, yy2 = np.meshgrid(seg4_x, y, sparse=True)
    seg2_xx = seg2_xx.reshape(slope_width, 1)
    seg4_xx = seg4_xx.reshape(slope_width, 1)
    # 高度赋值
    terrain.height_field_raw[:, :] = 0
    terrain.height_field_raw[seg1_x, :] = max_height
    terrain.height_field_raw[seg2_x, :] += (max_height * (seg2_xx - terrain.width / 16) / slope_width).astype(
        terrain.height_field_raw.dtype)
    terrain.height_field_raw[seg3_x, :] = 0
    terrain.height_field_raw[seg4_x, :] += (max_height * (seg4_xx - terrain.width * 9 / 16) / slope_width).astype(
        terrain.height_field_raw.dtype)
    terrain.height_field_raw[seg5_x, :] = max_height

    return terrain


def stairs_terrain_improved(terrain, step_width, step_height):
    '''
    stairs only exist along x-axis

    3 segs: stairs -> platform -> stairs
    '''
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    # width of the central platform
    platform_width = int(terrain.width / 8)
    # num of steps
    num_steps = int((7 * terrain.width / 8) // step_width)
    print(num_steps)
    num_steps_per_seg = int(num_steps / 2)
    # seg1: stairs(down)
    max_height = num_steps_per_seg * step_height
    height = max_height
    for i in range(num_steps_per_seg):
        terrain.height_field_raw[i * step_width:(i + 1) * step_width, :] = height
        height -= step_height
    # seg2: platform
    terrain.height_field_raw[int(7 * terrain.width / 16):int(9 * terrain.width / 16), :] = 0
    # seg3: stairs(up)
    height = step_height
    for i in range(num_steps_per_seg):
        terrain.height_field_raw[
        int(9 * terrain.width / 16) + i * step_width: int(9 * terrain.width / 16) + (i + 1) * step_width, :] = height
        height += step_height
    return terrain