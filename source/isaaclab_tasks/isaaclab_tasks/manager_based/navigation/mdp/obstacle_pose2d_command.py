# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.envs.mdp import UniformPose2dCommand, UniformPose2dCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ObstaclePose2dCommand(UniformPose2dCommand):
    """Command generator that generates pose commands based on the obstacle.

    This command generator determines the command position based on the position of the obstacle.
    The heading commands are either set to point towards the target or are sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.
    """

    cfg: ObstaclePose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: ObstaclePose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the obstacle object
        self.object: RigidObject = env.scene[cfg.object_name]

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        r = torch.empty(len(env_ids), device=self.device)
        # offset the position command by the current root position
        self.pos_command_w[env_ids, 0] = (
            2 * self.object.data.root_pos_w[env_ids, 0] - self.robot.data.root_pos_w[env_ids, 0]
        )
        self.pos_command_w[env_ids, 1] = (
            2 * self.object.data.root_pos_w[env_ids, 1] - self.robot.data.root_pos_w[env_ids, 1]
        )
        self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)


@configclass
class ObstaclePose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the obstacle-based 2D-pose command generator."""

    class_type: type = ObstaclePose2dCommand

    object_name: str = MISSING
    """Name of the obstacle object in the environment."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the sampled commands."""
