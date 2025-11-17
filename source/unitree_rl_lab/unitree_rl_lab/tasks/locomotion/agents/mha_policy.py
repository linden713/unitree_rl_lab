# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import field
from typing import Tuple

import contextlib
import io

import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from rsl_rl.modules import ActorCritic
from rsl_rl.storage.rollout_storage import RolloutStorage
from tensordict import TensorDict, TensorDictBase


def _build_observation_buffer(sample, num_steps, device):
    """Create a rollout storage buffer matching the nested structure of ``sample``."""

    if isinstance(sample, TensorDictBase):
        sub_buffers = {
            key: _build_observation_buffer(sample.get(key), num_steps, device)
            for key in sample.keys()
        }
        return TensorDict(
            sub_buffers,
            batch_size=[num_steps, *sample.batch_size],
            device=device,
        )

    if torch.is_tensor(sample):
        return torch.zeros(
            (num_steps, *sample.shape),
            device=device,
            dtype=sample.dtype,
        )

    raise TypeError(f"Unsupported observation leaf type: {type(sample)}")


class MHAEncoder(nn.Module):
    """CNN encoder for processing map scan data."""
    def __init__(
        self,
        d=64,
        h=16,
        proprio_dim=128,
        scan_size: Tuple[float, float] = (1.6, 1.0),
        resolution: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.attention_head = h
        size_x = float(scan_size[0])
        size_y = float(scan_size[1])
        x = torch.arange(-size_x / 2.0, size_x / 2.0 + resolution, resolution)
        y = torch.arange(-size_y / 2.0, size_y / 2.0 + resolution, resolution)

        self.grid_shape = (x.numel(), y.numel())
        self.map_points = self.grid_shape[0] * self.grid_shape[1]
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        self.register_buffer("grid_x", xx, persistent=False)
        self.register_buffer("grid_y", yy, persistent=False)

        # First CNN layer: 1 -> 16 channels, kernel_size=5, padding=2
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)

        # Second CNN layer: 16 -> (d-3) channels, kernel_size=5, padding=2
        self.conv2 = nn.Conv2d(16, d - 3, kernel_size=5, padding=2)

        # Proprioception projection layer
        self.proprio_proj = nn.Linear(proprio_dim, d)
        
        # 多头注意力
        self.mha = nn.MultiheadAttention(
            embed_dim=d, 
            num_heads=h, 
            batch_first=True,
            dropout=0.1
        )

    def forward(self, map_scans, proprioception):
        """Map height scans to a regular point cloud and extract features."""

        B = map_scans.shape[0]
        grid_x = self.grid_x.to(device=map_scans.device, dtype=map_scans.dtype)
        grid_y = self.grid_y.to(device=map_scans.device, dtype=map_scans.dtype)

        # Ensure input has shape (B, H, W)
        if map_scans.dim() == 2:
            heights = map_scans.view(B, *self.grid_shape)
        else:
            heights = map_scans
        xx_expanded = grid_x.unsqueeze(0).expand(B, -1, -1)
        yy_expanded = grid_y.unsqueeze(0).expand(B, -1, -1)
        zz = heights
        map_3d = torch.stack([xx_expanded, yy_expanded, zz], dim=-1)

        # Extract z-coordinates as a height map
        z_coords = map_3d[..., 2].unsqueeze(1)  # (B, 1, L, W)

        # CNN processing
        feats = F.relu(self.conv1(z_coords))  # (B, 16, L, W)
        feats = F.relu(self.conv2(feats))  # (B, d-3, L, W)

        # Reshape to (B, L*W, d-3)
        feats = feats.permute(0, 2, 3, 1).reshape(B, -1, self.d - 3)

        # Flatten 3D coordinates
        xyz_flat = map_3d.reshape(B, -1, 3)  # (B, L*W, 3)

        # Concatenate CNN features and 3D coordinates
        local_features = torch.cat([feats, xyz_flat], dim=-1)  # (B, L*W, d)

        proprio_emb = self.proprio_proj(proprioception).unsqueeze(1)

        # Multi-head attention: proprioception as query, local_features as key and value
        attn_output, _ = self.mha(
            query=proprio_emb,
            key=local_features, 
            value=local_features
        )
        
        # Squeeze the sequence dimension
        map_encoding = attn_output.squeeze(1)  # (B, d)
        
        return map_encoding


class ActorHead(nn.Module):

    def __init__(self, shared: MHAEncoder, proprio_dim: int, action_dim: int):
        super().__init__()
        self.shared = shared
        in_dim = shared.d + proprio_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ELU(),
            nn.Linear(256, 128),    nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, map_scans, proprio):
        map_encoding = self.shared(map_scans, proprio)      # (B, d)
        fused = torch.cat([map_encoding, proprio], dim=-1)  # (B, d + proprio)
        return self.mlp(fused)                              # Action mean (B, A)


class CriticHead(nn.Module):
    def __init__(self, shared: MHAEncoder, proprio_dim: int, critic_extra_dim: int):
        super().__init__()
        self.shared = shared
        in_dim = shared.d + proprio_dim + critic_extra_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ELU(),
            nn.Linear(256, 128),    nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, map_scans, proprio, critic_extra):
        map_encoding = self.shared(map_scans, proprio)              # (B, d)
        fused = torch.cat([map_encoding, proprio, critic_extra], dim=-1)
        return self.mlp(fused)                                      # Value (B, 1)


class MHAPolicy(ActorCritic):
    """Custom policy network based on CNN+MHA."""
    
    def __init__(
        self, 
        obs_space, 
        obs_groups,
        action_space_dim=12, 
        d=64,
        h=16,
        scan_size=(1.6, 1.0),
        scan_resolution=0.1,
        activation="elu",
        compile_networks=False,
        **kwargs,
    ):
        print("Initialization parameters:")
        for k, v in locals().items():
            if k != "self":
                print(f"  {k}: {v}")
        print("="*20)
        # 1. Call the parent constructor.
        # We need to call the parent's __init__. However, the parent class creates standard MLP networks,
        # which conflicts with our custom network.
        # One trick is to not call it first, and after defining our own network, manually call some of the parent's initializations.
        # Alternatively, we can first call the parent with fake parameters and then override the actor and critic.
        # Here we choose the latter as it is more conventional for inheritance.
        # We need to call the parent constructor. Since we are building a custom network,
        # we can pass dummy values for dimensions, as we will overwrite the actor and critic networks.
        silent_buffer = io.StringIO()
        with contextlib.redirect_stdout(silent_buffer):
            super().__init__(
                obs={"policy": torch.zeros(1, 1), "critic": torch.zeros(1, 1)},
                obs_groups=obs_groups,
                num_actions=action_space_dim,
                activation=activation,
                **kwargs
            )

        # Restore the correct obs_groups
        self.obs_groups = obs_groups
        self.d = d
        self.h = h
        self.scan_size = scan_size

        # (Removed profiling toggles and instrumentation.)

        # Dynamically get dimensions from the observation space
        policy_obs_space = obs_space[obs_groups["policy"][0]]
        critic_obs_space = obs_space[obs_groups["critic"][0]]
        self.proprio_dim = self._get_proprio_dim(policy_obs_space)
        self.critic_extra_dim = self._get_critic_extra_dim(critic_obs_space)

        # Pre-define keys for concatenation to avoid list creation in forward pass
        self.proprio_keys = [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos_rel",
            "joint_vel_rel",
            "last_action",
        ]
        self.critic_extra_keys = [
            "base_lin_vel",
            "joint_effort",
        ]
        # Ensure all keys exist in the observation space for safety
        assert all(key in policy_obs_space.keys() for key in self.proprio_keys)
        assert all(key in critic_obs_space.keys() for key in self.critic_extra_keys)

        self.MHA_encoder = MHAEncoder(
            d=d,
            h=h,
            proprio_dim=self.proprio_dim,
            scan_size=scan_size, 
            resolution=scan_resolution,
        )

        self.actor  = ActorHead(self.MHA_encoder,  self.proprio_dim, action_space_dim)
        self.critic = CriticHead(self.MHA_encoder, self.proprio_dim, self.critic_extra_dim)

        # Compile the actor and critic networks for better performance.
        # This is available from PyTorch 2.0 onwards.
        # As the MHA_encoder is a shared part of both actor and critic,
        # compiling them will also compile the shared encoder.
        if compile_networks:
            self.actor = torch.compile(self.actor)
            self.critic = torch.compile(self.critic)
        
    def act(self, obs, **kwargs):
        # This method is called during data collection.
        # It computes the action mean, updates the distribution, and samples an action.
        map_scans, proprioception = self._prepare_actor_inputs(obs)
        actions_mean = self.actor(map_scans, proprioception)
        self.update_distribution(actions_mean)
        return self.distribution.sample()
    
    def act_inference(self, obs, **kwargs):
        # This is used for deployment/evaluation.
        map_scans, proprioception = self._prepare_actor_inputs(obs)
        return self.actor(map_scans, proprioception)


    def evaluate(self, obs, actions=None, **kwargs):
        map_scans, proprioception, critic_extra = self._prepare_critic_inputs(obs)
        value = self.critic(map_scans, proprioception, critic_extra)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        This method is overridden to handle a key mismatch that can occur when
        loading a checkpoint saved from a non-compiled model into a compiled model, or vice-versa.
        The `torch.compile` function adds a `_orig_mod.` prefix to the keys. This method
        adds or removes that prefix to match the current model's state_dict.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.
        """
        model_has_prefix = any("_orig_mod" in k for k in self.state_dict())
        ckpt_has_prefix = any("_orig_mod" in k for k in state_dict)

        if model_has_prefix and not ckpt_has_prefix:
            # Case 1: Add prefix to checkpoint keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("actor.") or k.startswith("critic."):
                    parts = k.split(".", 1)
                    new_key = f"{parts[0]}._orig_mod.{parts[1]}"
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        elif not model_has_prefix and ckpt_has_prefix:
            # Case 2: Remove prefix from checkpoint keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if "_orig_mod" in k:
                    new_key = k.replace("._orig_mod", "")
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        return super().load_state_dict(state_dict, strict=strict)

    
    def get_actor_obs(self, obs):
        # Override to directly return the parsed observation dictionary
        return obs[self.obs_groups["policy"][0]]

    def get_critic_obs(self, obs):
        # Override to directly return the parsed observation dictionary
        return obs[self.obs_groups["critic"][0]]

    def update_distribution(self, actions_mean):
        # This is a helper function to create the action distribution, used by the parent's act method.
        if self.noise_std_type == "scalar":
            std = torch.clamp(self.std, min=1e-6).expand_as(actions_mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(actions_mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )
        self.distribution = torch.distributions.Normal(actions_mean, std)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _get_proprio_dim(self, policy_obs_space: TensorDictBase) -> int:
        """Calculate the total dimension of proprioceptive observations."""
        return sum(v.shape[-1] for k, v in policy_obs_space.items() if k != "height_scan")

    def _get_critic_extra_dim(self, critic_obs_space: TensorDictBase) -> int:
        """Calculate the extra observation dimension for the Critic."""
        return critic_obs_space["base_lin_vel"].shape[-1] + critic_obs_space["joint_effort"].shape[-1]

    def _prepare_inputs(self, obs_dict, proprio_keys, extra_keys=None):
        map_scans = obs_dict["height_scan"]
        proprioception = torch.cat([obs_dict[key] for key in proprio_keys], dim=-1)
        
        if extra_keys:
            extra_data = torch.cat([obs_dict[key] for key in extra_keys], dim=-1)
            return map_scans, proprioception, extra_data
        
        return map_scans, proprioception

    def _prepare_actor_inputs(self, obs):
        actor_obs_dict = self.get_actor_obs(obs)
        return self._prepare_inputs(actor_obs_dict, self.proprio_keys)

    def _prepare_critic_inputs(self, obs):
        critic_obs_dict = self.get_critic_obs(obs)
        return self._prepare_inputs(critic_obs_dict, self.proprio_keys, self.critic_extra_keys)


import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner
setattr(_rsl_on_policy_runner, "MHAPolicy", MHAPolicy)


@configclass
class MHAPolicyCfg(RslRlPpoActorCriticCfg):
    """Custom CNN+MHA policy configuration, extending RSL-RL's PPO policy configuration."""

    class_name: str = "MHAPolicy"
    init_noise_std: float = 1.0
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    d: int = 64  # MHA dimension
    h: int = 16  # Number of attention heads
    scan_size: Tuple[float, float] = (1.6, 1.0)  # Ray scan coverage (length, width)
    scan_resolution: float = 0.1
    compile_networks=True,



def _patch_rollout_storage():
    """Ensure RolloutStorage can handle plain dict observations."""

    original_add_transitions = RolloutStorage.add_transitions

    def add_transitions_patched(self, transition):
        if not isinstance(transition.observations, TensorDictBase):
            return original_add_transitions(self, transition)

        if not hasattr(self, "_custom_cnn_mha_buffer_ready"):
            self.observations = _build_observation_buffer(
                transition.observations,
                self.num_transitions_per_env,
                self.device,
            )
            self._custom_cnn_mha_buffer_ready = True

        transition.observations = transition.observations.clone(recurse=True)
        return original_add_transitions(self, transition)

    RolloutStorage.add_transitions = add_transitions_patched


_patch_rollout_storage()
