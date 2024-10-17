from pathlib import Path

import gym
import d4rl
import numpy as np
import torch
from tqdm import trange
import json
import torch.nn as nn
from src.iql import ImplicitQLearning, ImplicitQLearningMSE
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction, TwinQMSE
from src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy, DEFAULT_DEVICE
from ae_mse_util import LSTM_AE


def add_sa_history(dataset, history_length=5):
    # Extract relevant arrays
    observations = dataset['observations']
    actions = dataset['actions']
    
    # Get shapes
    num_samples, obs_dim = observations.shape
    _, act_dim = actions.shape
    
    # Concatenate observations and actions
    combined = np.concatenate([observations, actions], axis=1)
    
    # Create empty array for sa_hist
    sa_hist = np.zeros((num_samples, history_length, obs_dim + act_dim))
    
    # Fill sa_hist using rolling window
    for i in range(history_length):
        if i == 0:
            sa_hist[:, -1, :] = combined
        else:
            sa_hist[i:, -i-1, :] = combined[:-i]
    
    # Add new key to dataset
    dataset['sa_hist'] = sa_hist
    
    return dataset

def get_env_and_dataset(log, env_name, max_episode_steps):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    dataset = add_sa_history(dataset)
    for key in dataset.keys():
        print({key: dataset[key].shape})

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset


def main(args):
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir)/args.env_name, vars(args))
    log(f'Log dir: {log.dir}')

    env, dataset = get_env_and_dataset(log, args.env_name, args.max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    mse_dim = 1
    set_seed(args.seed, env=env)

    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, policy, args.max_episode_steps) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })

    #############################################################################
    ae_model = LSTM_AE(
        input_dim=obs_dim + act_dim,
        encoding_dim=16,
        h_dims=[32],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh()
    )
    ae_model.load_state_dict(torch.load("/home/azureuser/cloudfiles/code/Users/t-shandilyas/LSTM-AE/trained_models/halfcheetah-medium-v2_trial_2/lstm_ae_model_final.pt", weights_only=True))
    ae_model = ae_model.to(DEFAULT_DEVICE)
    ae_model.eval()
    ##############################################################################

    iql = ImplicitQLearningMSE(
        qf=TwinQMSE(obs_dim, act_dim, mse_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )


    for step in trange(args.n_steps):
        batch = sample_batch(dataset, args.batch_size)
        sa_hist = torchify(batch["sa_hist"])
        with torch.no_grad():
            _, ae_output = ae_model(sa_hist)
        mse_loss = nn.functional.mse_loss(ae_output, sa_hist, reduction='none').mean(dim=(1, 2), keepdim=True).squeeze(1)
        batch["mse_loss"] = mse_loss
        batch.pop("sa_hist")  
              
        iql.update(**batch)
        if (step+1) % args.eval_period == 0:
            eval_policy()

    torch.save(iql.state_dict(), log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    main(parser.parse_args())