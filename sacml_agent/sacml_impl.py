import copy
import math
from typing import Any, Optional, Sequence, Tuple, List, Union

import numpy as np
import torch
from torch.optim import Optimizer

from d3rlpy.gpu import Device
from d3rlpy.models.builders import (
    create_parameter,
    create_squashed_normal_policy,
    create_non_squashed_normal_policy,
)
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch import (
    Parameter,
    SquashedNormalPolicy,
    NonSquashedNormalPolicy,
)
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, torch_api, train_api, eval_api
from d3rlpy.algos.torch.ddpg_impl import DDPGBaseImpl

from d3rlpy.models.builders import (
    create_value_function,
)

from utility import EnsembleZFunction

class SACMLImpl(DDPGBaseImpl):
    r""" 
    SAC+ML for offline training, implemented with minimal changes upon 
    https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/torch/sac_impl.py
    """

    _policy: Optional[Union[SquashedNormalPolicy, NonSquashedNormalPolicy]]
    _targ_policy: Optional[Union[SquashedNormalPolicy, NonSquashedNormalPolicy]]
    _temp_learning_rate: float
    _temp_optim_factory: OptimizerFactory
    _initial_temperature: float
    _log_temp: Optional[Parameter]
    _temp_optim: Optional[Optimizer]
    # SAC+ML
    _baseline_learning_rate: float
    _baseline_optim_factory: OptimizerFactory
    _baseline_encoder_factory: EncoderFactory
    _alpha: float
    _target_entropy: float
    _squashed_normal_policy: bool
    _z_func: Optional[EnsembleZFunction]


    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        initial_temperature: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        # SAC+ML
        baseline_learning_rate: float,
        baseline_encoder_factory: EncoderFactory,
        baseline_optim_factory: OptimizerFactory,
        alpha: float,
        target_entropy: float,
        squashed_normal_policy: bool,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._temp_learning_rate = temp_learning_rate
        self._temp_optim_factory = temp_optim_factory
        self._initial_temperature = initial_temperature

        # initialized in build
        self._log_temp = None
        self._temp_optim = None

        # SAC+ML
        self._alpha = alpha
        self._target_entropy = target_entropy if target_entropy != None else self._action_size
        self._baseline_learning_rate = baseline_learning_rate
        self._baseline_optim_factory = baseline_optim_factory
        self._baseline_encoder_factory = baseline_encoder_factory
        self._squashed_normal_policy = squashed_normal_policy

    def build(self) -> None:
        self._build_temperature()
        self._build_baseline()
        super().build()
        self._build_temperature_optim()
        self._build_baseline_optim()

    def _build_actor(self) -> None:
        if self._squashed_normal_policy:
            self._policy = create_squashed_normal_policy(
                self._observation_shape,
                self._action_size,
                self._actor_encoder_factory,
            )
        else:
            self._policy = create_non_squashed_normal_policy(
                self._observation_shape,
                self._action_size,
                self._actor_encoder_factory,
                min_logstd=-5.0,
                max_logstd=0.0,
                use_std_parameter=True,
            )
    
    def _build_baseline(self) -> None:
        # we set the number of baseline function Z(s) to be consistent with number of Q(s,a)
        n_baselines = self._n_critics
        z_funcs = []
        for _ in range(n_baselines):
            z_funcs.append(create_value_function(self._observation_shape, self._baseline_encoder_factory))
        self._z_func = EnsembleZFunction(z_funcs)

    def _build_baseline_optim(self) -> None:
        assert self._z_func is not None
        self._baseline_optim = self._baseline_optim_factory.create(
            self._z_func.parameters(), lr=self._baseline_learning_rate
        )

    def _build_temperature(self) -> None:
        initial_val = math.log(self._initial_temperature)
        self._log_temp = create_parameter((1, 1), initial_val)

    def _build_temperature_optim(self) -> None:
        assert self._log_temp is not None
        self._temp_optim = self._temp_optim_factory.create(
            self._log_temp.parameters(), lr=self._temp_learning_rate
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._q_func is not None
        action, log_prob = self._policy.sample_with_log_prob(batch.observations)
        entropy = self._log_temp().exp() * log_prob
        q_t = self._q_func(batch.observations, action, "min")
        return (entropy - q_t).mean()

    @train_api
    @torch_api()
    def update_temp(
        self, batch: TorchMiniBatch
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self._temp_optim is not None
        assert self._policy is not None
        assert self._log_temp is not None

        self._temp_optim.zero_grad()

        with torch.no_grad():
            _, log_prob = self._policy.sample_with_log_prob(batch.observations)
            targ_temp = log_prob - self._target_entropy

        loss = -(self._log_temp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        # current temperature value
        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_temp
    

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action, log_prob = self._policy.sample_with_log_prob(
                batch.next_observations
            )
            entropy = self._log_temp().exp() * log_prob
            target = self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
            return target - entropy
        
    # SAC+ML and baseline training
    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._q_func is not None
        
        action, log_prob = self._policy.sample_with_log_prob(batch.observations)
        entropy = self._log_temp().exp() * log_prob

        q_t = self._q_func(batch.observations, action, "min")
        lam = self._alpha / (q_t.abs().mean()).detach()
        
        likelihood = -self._policy.dist(batch.observations).log_prob(batch.actions).mean()

        return lam * (entropy - q_t).mean() + likelihood 

    @train_api
    @torch_api()
    def update_baseline(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._baseline_optim is not None
        
        self._baseline_optim.zero_grad()
        
        z_tpn = self.compute_baseline_target(batch)
        loss = self.compute_baseline_loss(batch, z_tpn)
        
        loss.backward()
        self._baseline_optim.step()
        
        return loss.cpu().detach().numpy()
    
    def compute_baseline_loss(
        self, batch: TorchMiniBatch, z_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._z_func is not None
        target = z_tpn
        assert target.ndim == 2

        with torch.no_grad():
            distr = self._policy.dist(batch.observations)
            log_prob = distr.log_prob(batch.actions).detach()
            y = batch.rewards + self._gamma * target * (1 - batch.terminals)
        
        loss = 0
        for z_func in self._z_func._z_funcs:
            baseline = z_func.forward(batch.observations)
            value = log_prob + baseline
            loss += torch.nn.functional.mse_loss(value, y.detach(), reduction="none").mean()
        
        return loss

    def compute_baseline_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._z_func is not None
        
        with torch.no_grad():
            
            action, log_prob = self._policy.sample_with_log_prob(
                batch.next_observations
            )
            entropy = self._log_temp().exp() * log_prob
            
            target_baseline = self._z_func.compute_target(
                batch.next_observations,
                reduction="min",
            )
            target = log_prob + target_baseline
            return target - entropy
        