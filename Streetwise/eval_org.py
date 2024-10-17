import numpy as np
import d4rl
import torch
import gym
from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
import torch.nn as nn
import matplotlib.pyplot as plt
import random

random.seed(21)  
np.random.seed(21)


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x

def generate_value_sequence(initial, final, num_points):
    if num_points < 2:
        raise ValueError("The number of points must be at least 2")
    if initial == final:
        raise ValueError("Initial and final values must be different")
    
    step = (final - initial) / (num_points - 1)
    return [initial + i * step for i in range(num_points)]


def generate_gravity_variation(num_steps, variation_type='burst', base_gravity=-9.81, **kwargs):
    """
    Generate various patterns of gravity variation.
    
    :param num_steps: Number of steps in the episode
    :param variation_type: Type of gravity variation ('burst', 'sinusoidal', 'random_walk', 'step_changes', 'intermittent_bursts', 'constant_with_jumps', 'constant_with_noise_and_decay')
    :param base_gravity: Base gravity value
    :param kwargs: Additional parameters specific to each variation type
    :return: Array of gravity values for each step
    """
    t = np.arange(num_steps)
    gravity_variation = np.full(num_steps, base_gravity)
    
    if variation_type == 'burst':
        burst_frequency = kwargs.get('burst_frequency', 0.01)
        burst_amplitude = kwargs.get('burst_amplitude', 5)
        noise_scale = kwargs.get('noise_scale', 0.5)
        
        periodic_component = burst_amplitude * np.sin(2 * np.pi * burst_frequency * t)
        noise = np.random.normal(0, noise_scale, num_steps)
        gravity_variation += periodic_component + noise
    
    elif variation_type == 'sinusoidal':
        amplitude = kwargs.get('amplitude', 2)
        frequency = kwargs.get('frequency', 0.005)
        
        gravity_variation += amplitude * np.sin(2 * np.pi * frequency * t)
    
    elif variation_type == 'random_walk':
        step_size = kwargs.get('step_size', 0.1)
        
        random_walk = np.cumsum(np.random.normal(0, step_size, num_steps))
        gravity_variation += random_walk
    
    elif variation_type == 'step_changes':
        num_changes = kwargs.get('num_changes', 5)
        max_change = kwargs.get('max_change', 3)
        
        change_points = np.sort(np.random.choice(num_steps, num_changes, replace=False))
        changes = np.random.uniform(-max_change, max_change, num_changes)
        
        for point, change in zip(change_points, changes):
            gravity_variation[point:] += change
    
    elif variation_type == 'intermittent_bursts':
        burst_amplitude = kwargs.get('burst_amplitude', 5)
        burst_duration = kwargs.get('burst_duration', 50)
        num_bursts = kwargs.get('num_bursts', 3)
        
        burst_starts = np.sort(np.random.choice(num_steps - burst_duration, num_bursts, replace=False))
        for start in burst_starts:
            gravity_variation[start:start+burst_duration] += burst_amplitude * np.sin(np.linspace(0, 2*np.pi, burst_duration))
    
    elif variation_type == 'constant_with_jumps':
        jump_amplitude = kwargs.get('jump_amplitude', 5)
        jump_duration = kwargs.get('jump_duration', 10)
        num_jumps = kwargs.get('num_jumps', 5)
        
        jump_starts = np.sort(np.random.choice(num_steps - jump_duration, num_jumps, replace=False))
        for start in jump_starts:
            gravity_variation[start:start+jump_duration] += jump_amplitude
    
    elif variation_type == 'step_like_pattern':
        num_steps = kwargs.get('num_steps', 1000)
        initial_value = kwargs.get('initial_value', -10)
        second_value = kwargs.get('second_value', -5)
        third_value = kwargs.get('third_value', -20)
        final_value = kwargs.get('final_value', -15)
        first_step = kwargs.get('first_step', 300)
        second_step = kwargs.get('second_step', 600)
        third_step = kwargs.get('third_step', 900)
        noise_scale = kwargs.get('noise_scale', 1)
        
        gravity_variation = np.zeros(num_steps)
        gravity_variation[:first_step] = initial_value
        gravity_variation[first_step:second_step] = second_value
        gravity_variation[second_step:third_step] = third_value
        gravity_variation[third_step:] = generate_value_sequence(third_value, final_value, len(gravity_variation[third_step:]))
        
        # Add some noise to make it look more natural
        noise = np.random.normal(0, noise_scale, num_steps)
        gravity_variation += noise
        
        # Smooth transitions
        window_size = 5
        gravity_variation = np.convolve(gravity_variation, np.ones(window_size)/window_size, mode='same')
        
        # Ensure the final value is reached
        gravity_variation[-1] = final_value
    
    else:
        raise ValueError(f"Unknown variation type: {variation_type}")
    
    return gravity_variation

def evaluate_policy(env, policy, max_episode_steps, deterministic=True):
    obs = env.reset()
    total_reward = 0.
    gravity_values = generate_gravity_variation(max_episode_steps, variation_type='step_like_pattern')
    
    for step in range(max_episode_steps):
        # Update gravity for the current step
        env.sim.model.opt.gravity[2] = gravity_values[step]
        
        with torch.no_grad():
            action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward, gravity_values

def eval_policy(env, env_name, policy, max_episode_steps, n_eval_episodes):
    eval_returns = []
    all_gravity_values = []
    
    for _ in range(n_eval_episodes):
        total_reward, gravity_values = evaluate_policy(env, policy, max_episode_steps)
        eval_returns.append(total_reward)
        all_gravity_values.extend(gravity_values)
    
    eval_returns = np.array(eval_returns)
    all_gravity_values = np.array(all_gravity_values)
    normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0
    
    results = {
        "return_mean": eval_returns.mean(),
        "return_std": eval_returns.std(),
        "normalized_return_mean": normalized_returns.mean(),
        "normalized_return_std": normalized_returns.std(),
        "gravity_values": all_gravity_values,
        "max_episode_steps": max_episode_steps,
        "n_eval_episodes": n_eval_episodes
    }
    
    return results

env_name = 'halfcheetah-medium-v2'
model_path = '/home/azureuser/cloudfiles/code/Users/t-shandilyas/IQL_RESULTS/halfcheetah-medium-v2_trial_1/halfcheetah-medium-v2/10-06-24_14.57.03_unrg/final.pt'

env = gym.make(env_name)
deterministic_policy = False

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(f"Environment: {env_name}")
print(f'Observation dim: {obs_dim}, Action dim: {act_dim}')

args = {
    'hidden_dim': 256,
    'n_hidden': 2,
    'learning_rate': 3e-4,
    'n_steps': 1000000,
    'tau': 0.7,
    'beta': 3.0,
    'alpha': 0.005,
    'discount': 0.99
}

if deterministic_policy:
    policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args['hidden_dim'], n_hidden=args['n_hidden'])
else:
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args['hidden_dim'], n_hidden=args['n_hidden'])

iql = ImplicitQLearning(
    qf=TwinQ(obs_dim, act_dim, hidden_dim=args['hidden_dim'], n_hidden=args['n_hidden']),
    vf=ValueFunction(obs_dim, hidden_dim=args['hidden_dim'], n_hidden=args['n_hidden']),
    policy=policy,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=args['learning_rate']),
    max_steps=args['n_steps'],
    tau=args['tau'],
    beta=args['beta'],
    alpha=args['alpha'],
    discount=args['discount']
)

# load weights to iql learning model
iql.load_state_dict(torch.load(model_path))

# evaluate the policy
results = eval_policy(env, env_name, iql.policy, max_episode_steps=1000, n_eval_episodes=10)


# Plot concatenated gravity values
plt.figure(figsize=(15, 6))
plt.plot(results['gravity_values'])
plt.title("Gravity Variation Across All Evaluation Episodes")
plt.xlabel("Steps")
plt.ylabel("Gravity")

# Add vertical lines to separate episodes
for i in range(1, results['n_eval_episodes']):
    plt.axvline(x=i * results['max_episode_steps'], color='r', linestyle='--', alpha=0.5)

plt.savefig(f"gravity_variation_{env_name}_org_eval.png")
plt.show()

print(f"Mean Return: {results['return_mean']:.2f} ± {results['return_std']:.2f}")
print(f"Mean Normalized Return: {results['normalized_return_mean']:.2f} ± {results['normalized_return_std']:.2f}")