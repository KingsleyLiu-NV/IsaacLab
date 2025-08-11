# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_uniform_angular(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    radius_range: tuple[float, float],
    angle_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset the asset root state to a random position within a sector of a ring area defined by the radius and angle ranges.

    This function randomizes the root position of the asset.

    * It samples the root position from the given angular ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It keeps the root orientation unchanged.
    * It keeps the root velocity unchanged.

    The function takes tuples of the form ``(min, max)`` for the radius and angle ranges.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [radius_range, angle_range]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 2), device=asset.device)

    rand_x = rand_samples[:, 0:1] * torch.cos(rand_samples[:, 1:2])
    rand_y = rand_samples[:, 0:1] * torch.sin(rand_samples[:, 1:2])
    rand_pos = torch.cat([rand_x, rand_y], dim=-1)

    positions = root_states[:, 0:2] + env.scene.env_origins[env_ids, 0:2] + rand_pos
    orientations = root_states[:, 2:7]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
