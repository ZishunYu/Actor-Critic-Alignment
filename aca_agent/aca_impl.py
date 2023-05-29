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

class ACAImpl(DDPGBaseImpl):
    r""" 
    SAC+ACA for online training, implemented with minimal changes upon 
    https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/torch/sac_impl.py
    """

    _policy: Optional[Union[SquashedNormalPolicy, NonSquashedNormalPolicy]]
    _targ_policy: Optional[Union[SquashedNormalPolicy, NonSquashedNormalPolicy]]
    _temp_learning_rate: float
    _temp_optim_factory: OptimizerFactory
    _initial_temperature: float
    _log_temp: Optional[Parameter]
    _temp_optim: Optional[Optimizer]
    # SAC+ACA
    _baseline_learning_rate: float
    _baseline_optim_factory: OptimizerFactory
    _baseline_encoder_factory: EncoderFactory
    _beta: float
    _target_entropy: float
    _interpolation: tuple
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
        # SAC+ACA
        baseline_learning_rate: float,
        baseline_encoder_factory: EncoderFactory,
        baseline_optim_factory: OptimizerFactory,
        beta: float,
        target_entropy: float,
        interpolation: tuple,
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

        # SAC+ACA
        self._beta = beta
        self._target_entropy = target_entropy if target_entropy != None else self._action_size
        self._baseline_learning_rate = baseline_learning_rate
        self._baseline_optim_factory = baseline_optim_factory
        self._baseline_encoder_factory = baseline_encoder_factory
        self._squashed_normal_policy = squashed_normal_policy
        self._interpolation = interpolation
        self._weight = self._interpolation[0]

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
    
    # SAC+ACA
    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()
        
        loss = self.compute_actor_loss(batch)
        
        self._actor_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 0.25)
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._q_func is not None
        action, log_prob = self._policy.sample_with_log_prob(batch.observations)
        entropy = self._log_temp().exp() * log_prob
        
        distr = self._reference_policy.dist(batch.observations)
        log_pi_0 = distr.log_prob(action)
        log_pi_0 = self.clip_log_pi(distr, log_pi_0, keep_grad=True) # clipped log_pi_0 value while keep the gradient direction
        
        r_t = self._q_func(batch.observations, action, "min") # R_\phi, see Eq. (TODO)
        q_t = r_t +  log_pi_0
        
        return (entropy - q_t).mean()

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        
        target = q_tpn
        assert target.ndim == 2

        td_sum = torch.tensor(
            0.0, dtype=torch.float32, device=batch.observations.device
        )
        
        # compute \log_\pi_0
        with torch.no_grad():
            distr = self._reference_policy.dist(batch.observations)
            log_pi_0 = distr.log_prob(batch.actions)
            log_pi_0 = self.clip_log_pi(distr, log_pi_0, keep_grad=False)
            
        for q_func in self._q_func._q_funcs:
            # compute R_\phi(s, a) + \log \pi_0 (s, a)
            r_t = q_func.forward(batch.observations, batch.actions)
            q_t = r_t + log_pi_0 

            # compute TD errors
            y = batch.rewards + self._gamma * target * (1 - batch.terminals)
            loss = torch.nn.functional.mse_loss(q_t, y.detach(), reduction="none")
            td_sum += loss.mean()
        
        return td_sum
        
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._targ_q_func is not None
        
        with torch.no_grad():
            action, log_prob = self._policy.sample_with_log_prob(
                batch.next_observations
            )
            entropy = self._log_temp().exp() * log_prob
            
            target_r = self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )

            distr = self._reference_policy.dist(batch.next_observations)
            log_pi_0 = distr.log_prob(action)
            log_pi_0 = self.clip_log_pi(distr, log_pi_0, keep_grad=False) # clipping

            # compute target, i.e. R_\phi(s', a') + log_pi_0(s', a')
            target = target_r + log_pi_0 
            return target - entropy

    def clip_log_pi(
        self, 
        distr,
        log_pi: torch.Tensor,
        keep_grad: bool,
        ) -> torch.Tensor:
        # we clip the \log\pi_0 term as it may become too large in absolute value, see Appendix A.TODO for details
        if self._squashed_normal_policy:
            with torch.no_grad():
                clip_at_l = distr._log_prob_from_raw_y(distr._mean - self._beta * distr.std)
                clip_at_r = distr._log_prob_from_raw_y(distr._mean + self._beta * distr.std)
                clip_at = torch.minimum(clip_at_l, clip_at_r)
            clipper = torch.nn.Softplus()
            if keep_grad:
                return self._weight * (clipper(log_pi - clip_at) + clip_at)
            else:
                return self._weight * (clipper(log_pi - clip_at) + clip_at).detach()
        else:
            with torch.no_grad():
                clip_at_l = distr.log_prob(distr._mean - self._beta * distr.std)
                clip_at_r = distr.log_prob(distr._mean + self._beta * distr.std)
                assert (clip_at_l.mean() - clip_at_r.mean()).abs() < 1e-2 # non-squash Gaussian is symmetric
                clip_at = torch.minimum(clip_at_l, clip_at_r)
            clipper = torch.nn.Softplus()
            if keep_grad:
                return self._weight * (clipper(log_pi - clip_at) + clip_at)
            else:
                return self._weight * (clipper(log_pi - clip_at) + clip_at).detach()