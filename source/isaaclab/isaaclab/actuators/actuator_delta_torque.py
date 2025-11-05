# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Delta torque actuator with neural network enhancement."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.types import ArticulationActions

from .actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .actuator_cfg import DeltaTorqueActuatorCfg


class DeltaTorqueActuator(ActuatorBase):
    """Delta torque actuator with neural network enhancement.

    This actuator combines PD control with neural network predictions to generate joint torques.
    It computes base torques using PD control and adds delta torques predicted by a neural network
    for specified joints only.

    The neural network takes as input:
    - Current joint positions (shape: num_joints)
    - Current joint velocities (shape: num_joints)
    - Position commands (shape: num_joints)

    And outputs delta torques for a subset of joints (typically the valid/controllable joints).
    """

    cfg: DeltaTorqueActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DeltaTorqueActuatorCfg, *args, **kwargs):
        """Initialize the delta torque actuator.

        Args:
            cfg: Configuration for the delta torque actuator.
            *args: Variable length argument list passed to the base class.
            **kwargs: Keyword arguments passed to the base class.
        """
        # Initialize base actuator
        super().__init__(cfg, *args, **kwargs)

        # Load neural network model
        self._load_network()

        # Create mapping for valid joints that receive delta torques
        self._setup_joint_mapping()

        # Storage for global joint data (will be set by articulation)
        self._global_joint_pos = None
        self._global_joint_vel = None
        self._global_joint_pos_target = None

    def reset(self, env_ids: Sequence[int]):
        """Reset the actuator state.

        Args:
            env_ids: List of environment IDs to reset.
        """
        pass  # No internal state to reset for this actuator

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Compute the actuator torques.

        Args:
            control_action: The joint action instance comprising desired joint positions,
                joint velocities and feed-forward joint efforts.
            joint_pos: Current joint positions. Shape is (num_envs, num_joints).
            joint_vel: Current joint velocities. Shape is (num_envs, num_joints).

        Returns:
            The updated control action with computed joint efforts.
        """
        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        # calculate the base PD torques
        base_torques = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts

        # compute delta torques using neural network (requires global joint data)
        if self._global_joint_pos is not None:
            delta_torques = self._compute_delta_torques(
                self._global_joint_pos, self._global_joint_vel, self._global_joint_pos_target
            )

            # start with base PD torques
            self.computed_effort = base_torques
            # apply delta torques only to valid joints
            self.computed_effort[:, self._valid_joint_indices] += delta_torques * self.cfg.action_scale
        else:
            self.computed_effort = base_torques

        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        # Update control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action

    def set_global_joint_data(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, joint_pos_target: torch.Tensor):
        """Set global joint data for neural network computation.

        Args:
            joint_pos: All joint positions. Shape: (num_envs, all_joints).
            joint_vel: All joint velocities. Shape: (num_envs, all_joints).
            joint_pos_target: All joint position targets. Shape: (num_envs, all_joints).
        """
        self._global_joint_pos = joint_pos
        self._global_joint_vel = joint_vel
        self._global_joint_pos_target = joint_pos_target

    def _load_network(self):
        """Load the neural network model from file."""
        self.network = torch.jit.load(self.cfg.network_file, map_location=self._device)
        self.network.eval()

    def _setup_joint_mapping(self):
        """Setup mapping for joints that receive delta torques."""
        if self.cfg.valid_joint_names is not None:
            valid_indices = [
                self.joint_names.index(name) for name in self.cfg.valid_joint_names if name in self.joint_names
            ]

            self._valid_joint_indices = torch.tensor(valid_indices, device=self._device, dtype=torch.long)
            self._num_valid_joints = len(valid_indices)
        else:
            self._valid_joint_indices = torch.arange(self.num_joints, device=self._device, dtype=torch.long)
            self._num_valid_joints = self.num_joints

    def _compute_delta_torques(
        self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, joint_pos_command: torch.Tensor
    ) -> torch.Tensor:
        """Compute delta torques using the neural network.

        Args:
            joint_pos: Current joint positions. Shape: (num_envs, num_joints).
            joint_vel: Current joint velocities. Shape: (num_envs, num_joints).
            joint_pos_command: Position commands. Shape: (num_envs, num_joints).

        Returns:
            Delta torques for valid joints. Shape: (num_envs, num_valid_joints).
        """
        # prepare network input: concatenate pos, vel, command
        network_input = torch.cat([joint_pos, joint_vel, joint_pos_command], dim=-1)

        # run neural network inference
        with torch.no_grad():
            return self.network(network_input)
