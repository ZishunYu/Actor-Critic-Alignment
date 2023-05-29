import argparse
import copy
import glob

import torch
import d3rlpy
import gym, d4rl

from utility import get_config, critic_init, buffer_init
from aca_agent.aca import ACA


def main(args):
    
    # load dataset and environment
    if "antmaze" in args.dataset:
        dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    else:
        dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    eval_env = gym.make(args.dataset)

    # set random seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    # load configs
    cfgs = get_config("configs/online", args.dataset)
    squashed_normal_policy = cfgs['squashed_normal_policy']
    n_trajs = cfgs['n_trajs']
    beta = cfgs['beta']
    interpolation = cfgs['interpolation']
    target_entropy = cfgs['target_entropy_coef'] * env.action_space.shape[0]

    # load pretrained model
    if "antmaze" in args.dataset:
        model_path = glob.glob(f"d3rlpy_logs/SAC+ML/{args.dataset}/{args.seed}/*/model_500000.pt")[0]
        reward_scaler = d3rlpy.preprocessing.reward_scalers.ConstantShiftRewardScaler(shift=-1)
    else:
        model_path = glob.glob(f"d3rlpy_logs/SAC+ML/{args.dataset}/{args.seed}/*/model_500000.pt")[0]
        reward_scaler = None
        
    # create ACA instance on top of SAC base
    aca = ACA(batch_size=256,
                actor_learning_rate=3e-4,
                critic_learning_rate=3e-4,
                temp_learning_rate=3e-4,
                beta=beta,
                interpolation=interpolation,
                target_entropy=target_entropy,
                squashed_normal_policy=squashed_normal_policy,
                reward_scaler=reward_scaler,
                use_gpu=args.gpu, 
                n_critics=2,)
    aca.build_with_env(env)
    aca.load_model(model_path)
    
    
    # reset aca temperature to 0 
    aca._impl._log_temp.state_dict()['_parameter'].copy_(torch.Tensor([0.0]))

    # keep a frozen reference policy \pi_0, see Eq. (TODO)
    aca._impl._reference_policy = copy.deepcopy(aca._impl._policy)

    # initialize R with Z, see Eq. (TODO) in our paper
    critic_init(aca, env, dataset)

    # initialize replay buffer for experience replay
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=int(1e6), env=env) 
    buffer_init(buffer, n_trajs, dataset)

    # start online training
    aca.fit_online(env,
                   buffer,
                   eval_env=eval_env,
                   n_steps=100000,
                   n_steps_per_epoch=1000,
                   update_interval=1,
                   update_start_step=0,
                   save_interval=100,
                   experiment_name=f"ACA/{args.dataset}/{args.seed}/exp",
                   with_timestamp=False,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    main(args)