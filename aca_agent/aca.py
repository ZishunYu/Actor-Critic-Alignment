import gym
import torch
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
)

from d3rlpy.argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.base import AlgoBase
from d3rlpy.dataset import TransitionMiniBatch


from .aca_impl import ACAImpl
    

class ACA(AlgoBase):
    r""" 
    SAC+ACA for online training, implemented with minimal changes upon 
    https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/sac.py
    Args:
        baseline_learning_rate (float): learning rate for the baseline Z(s) function.
        baseline_optim_factory (d3rlpy.models.optimizers.OptimizerFactory): optimizer factory for the actor.
        alpha: balancing weight of the maximum-loglikelihood regularizer.
        See https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/sac.py for the rest arguments.
    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _temp_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _initial_temperature: float
    _use_gpu: Optional[Device]
    _impl: Optional[ACAImpl]
    # additional attributes for SAC+ACA
    _baseline_learning_rate: float
    _baseline_encoder_factory: EncoderFactory
    _baseline_optim_factory: OptimizerFactory
    _beta: float
    _target_entropy: float
    _squashed_normal_policy: bool
    _interpolation: tuple

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        initial_temperature: float = 1.0,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[ACAImpl] = None,
        # additional SACML arguments
        baseline_learning_rate: float = 3e-4,
        baseline_encoder_factory: EncoderArg = "default",
        baseline_optim_factory: OptimizerFactory = AdamFactory(),
        beta: float = 1.0,
        target_entropy: float = None,
        interpolation: tuple = (1, 100, 1), 
        squashed_normal_policy: bool = True,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._initial_temperature = initial_temperature
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl
        # SAC+ACA
        self._baseline_learning_rate = baseline_learning_rate
        self._baseline_optim_factory = baseline_optim_factory
        self._baseline_encoder_factory = check_encoder(baseline_encoder_factory)
        self._beta = beta
        self._target_entropy = target_entropy
        self._interpolation = interpolation
        self._squashed_normal_policy = squashed_normal_policy

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = ACAImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            initial_temperature=self._initial_temperature,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            # SAC+ACA
            baseline_learning_rate = self._baseline_learning_rate, 
            baseline_optim_factory = self._baseline_optim_factory, 
            baseline_encoder_factory = self._baseline_encoder_factory, 
            beta = self._beta, 
            target_entropy = self._target_entropy,
            interpolation = self._interpolation, 
            squashed_normal_policy = self._squashed_normal_policy,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        # lagrangian parameter update for SAC temperature
        if self._temp_learning_rate > 0:
            temp_loss, temp = self._impl.update_temp(batch)
            metrics.update({"temp_loss": temp_loss, "temp": temp})

        critic_loss = self._impl.update_critic(batch)
        metrics.update({"critic_loss": critic_loss})

        actor_loss = self._impl.update_actor(batch)
        metrics.update({"actor_loss": actor_loss})

        self._impl.update_critic_target()
        self._impl.update_actor_target()

        self._update_interpolation()
        metrics.update({"interpolation_weight": self._impl._weight})

        return metrics
    
    def _update_interpolation(self, ):
        assert min(self._interpolation[0], self._interpolation[1]) > 0
        # hard-coded 1000 steps per episode, TODO: replace 1000 with n_steps_per_ep.
        self._impl._weight = max(self._impl._weight-self._interpolation[0]/(self._interpolation[1]*1000), self._interpolation[2]) 

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS