import argparse
import d3rlpy
from sklearn.model_selection import train_test_split
from sacml_agent.sacml import SACML
from utility import get_config

def main(args):

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    _, test_episodes = train_test_split(dataset, test_size=0.2)
    
    # load configs
    cfgs = get_config("configs/offline", args.dataset)
    alpha = cfgs['alpha']
    squashed_normal_policy = cfgs['squashed_normal_policy']
    target_entropy = cfgs['target_entropy_coef'] * env.action_space.shape[0]

    
    sac = SACML(batch_size=256,
                actor_learning_rate=3e-4,
                critic_learning_rate=3e-4,
                temp_learning_rate=3e-4,
                baseline_learning_rate=3e-4,
                use_gpu=args.gpu,
                alpha=alpha,
                target_entropy=target_entropy,
                squashed_normal_policy=squashed_normal_policy,
                n_critics=2,)
    
    sac.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=1000,
            save_interval=100,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
            },
            experiment_name=f"SAC+ML/{args.dataset}/{args.seed}/exp",
            with_timestamp=False,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    main(args)