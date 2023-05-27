from typing import List, Optional, Union, cast

import yaml
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from d3rlpy.models.torch.v_functions import ValueFunction
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.torch_utility import TorchMiniBatch
from d3rlpy.torch_utility import hard_sync

# =============================================
# Build emsembled baseline function
# =============================================

def _reduce_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    if reduction == "min":
        return y.min(dim=dim).values
    elif reduction == "max":
        return y.max(dim=dim).values
    elif reduction == "mean":
        return y.mean(dim=dim)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError

class EnsembleZFunction(nn.Module):  # type: ignore
    _z_funcs: nn.ModuleList

    def __init__(
        self,
        z_funcs: List[ValueFunction],
    ):
        super().__init__()
        self._z_funcs = nn.ModuleList(z_funcs)

    def forward(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        values = []
        for z_func in self._z_funcs:
            values.append(z_func(x).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, reduction))
    
    def _compute_target(
        self,
        x: torch.Tensor,
        reduction: str = "min",
    ) -> torch.Tensor:
        values_list: List[torch.Tensor] = []
        for z_func in self._z_funcs:
            target = z_func(x)
            values_list.append(target.reshape(1, x.shape[0], -1))

        values = torch.cat(values_list, dim=0)
        return _reduce_ensemble(values, reduction)
    
    def compute_target(
        self,
        x: torch.Tensor,
        reduction: str = "min",
    ) -> torch.Tensor:
        return self._compute_target(x, reduction)
    
    @property
    def z_funcs(self) -> nn.ModuleList:
        return self._z_funcs



# =============================================
# Lookup configuration file
# =============================================
def get_config(prefix, dataset):
    if "antmaze" in dataset:
        path = prefix + "/antmaze.yaml"
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
    elif "halfcheetah" in dataset or "hopper" in dataset or "walker2d" in dataset:
        if "-random-v2" in dataset:
            path = prefix + "/mjc_r.yaml"
        elif "-medium-v2" in dataset:
            path = prefix + "/mjc_m.yaml"
        elif "-medium-replay-v2" in dataset:
            path = prefix + "/mjc_mr.yaml"
        elif "-medium-expert-v2" in dataset:
            path = prefix + "/mjc_me.yaml"
        elif "-expert-v2" in dataset:
            path = prefix + "/mjc_e.yaml"
        else:
            raise "unreconganized dataset"
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        raise "no configuration for specified dataset"
    return cfg



# =============================================
# online critic initialization
# =============================================
def critic_init(aca, env, dataset):
    observation_size = env.observation_space._shape[0]
    for z_fn, q_fn in zip(aca._impl._z_func._z_funcs, aca._impl._q_func._q_funcs):
        state_dict = z_fn.state_dict()
        with torch.no_grad():
            # load layers except the first layer's weight
            q_fn._encoder._fcs[0].bias.copy_(state_dict['_encoder._fcs.0.bias'])
            q_fn._encoder._fcs[1].weight.copy_(state_dict['_encoder._fcs.1.weight'])
            q_fn._encoder._fcs[1].bias.copy_(state_dict['_encoder._fcs.1.bias'])
            q_fn._fc.weight.copy_(state_dict['_fc.weight'])
            q_fn._fc.bias.copy_(state_dict['_fc.bias'])
            # load first layers with only the state-dependent-part and set the reset with zeros
            state_weights = q_fn._encoder._fcs[0].weight[:, :observation_size]
            action_weights = q_fn._encoder._fcs[0].weight[:, observation_size:]
            state_weights.copy_(state_dict['_encoder._fcs.0.weight'][:, :observation_size])
            action_weights.copy_(torch.zeros(action_weights.shape))

        # assert the initialization is correct
        random_episode = dataset.episodes[np.random.randint(len(dataset.episodes))]
        random_transition = random_episode[np.random.randint(len(random_episode))]
        random_state, random_action = torch.Tensor(random_transition.observation).to(aca._impl._log_temp.data.device).unsqueeze(0), \
                                        torch.Tensor(random_transition.action).to(aca._impl._log_temp.data.device).unsqueeze(0)
        assert z_fn(random_state) == q_fn(random_state, random_action)
    # copy the target network
    hard_sync(aca._impl._targ_q_func, aca._impl._q_func)


# =============================================
# online buffer initialization
# =============================================
def buffer_init(buffer, N, dataset):
    if N!= 0:
        r = []
        for episode in tqdm.tqdm(dataset.episodes):
            if len(episode) > 0:
                batch = TransitionMiniBatch([tran for tran in episode])
                batch = TorchMiniBatch(batch, device="cuda")
                r.append(batch.rewards.sum().item())
        top_N_idx = np.argsort(r)[-N:]
        
        for idx in top_N_idx:
            buffer.append_episode(dataset.episodes[idx])