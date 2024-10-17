import numpy as np
import d4rl
import torch
import gym
from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
import torch.nn as nn
from ae_utils import AEOpexCellStateModel, LSTM_AE
import matplotlib.pyplot as plt
import random
import math
from scipy.signal import savgol_filter
import argparse
import sys, ast

random.seed(21)  
np.random.seed(21)

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def torchify(x):
    if isinstance(x, np.ndarray):
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
        initial_value = kwargs.get('initial_value', 1)
        second_value = kwargs.get('second_value', 3)
        third_value = kwargs.get('third_value', 5)
        final_value = kwargs.get('final_value', 3)
        first_step = kwargs.get('first_step', 100)
        second_step = kwargs.get('second_step', 400)
        third_step = kwargs.get('third_step', 700)
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

def generate_wind_variation(num_steps, base_wind=[0., 0., 0.], **kwargs):
    """
    Generate sinusoidal wind variation for the episode. There should be one complete cycle in the episode.
    
    :param num_steps: Number of steps in the episode
    :param base_wind: Base wind speed [x, y, z]
    :param kwargs: Additional parameters for wind variation
    :return: Array of wind speed values for each step (shape: num_steps x 3)
    """
    t = np.arange(num_steps)
    wind_variation = np.full((num_steps, 3), base_wind)
    
    amplitude = kwargs.get('amplitude', 5)
    num_cycles = kwargs.get('num_cycles', 1)

    # Generate sinusoidal wind variation for x axis only
    wind_variation[:, 0] += (amplitude + amplitude * np.sin(2 * np.pi * num_cycles * t / num_steps))

    return wind_variation
    

def generate_viscosity_variation(num_steps, base_viscosity=1.0, num_spikes=4, spike_duration=20, plateau_duration=100, max_spike_amplitude=2):
    """
    Generate viscosity variation with spikes and plateaus.
    
    :param num_steps: Number of steps in the episode
    :param base_viscosity: Base viscosity value
    :param num_spikes: Number of spikes in the variation (default: 4)
    :param spike_duration: Duration of the spike rise in steps (default: 20)
    :param plateau_duration: Duration of plateau after each spike (default: 50)
    :param max_spike_amplitude: Maximum amplitude of spikes (default: 0.5)
    :return: Array of viscosity values for each step
    """
    viscosity_variation = np.full(num_steps, base_viscosity)
    
    # Generate spikes and plateaus
    event_duration = spike_duration + plateau_duration
    available_start_positions = num_steps - event_duration
    
    if available_start_positions < num_spikes:
        raise ValueError("Not enough steps for the specified number of spikes and their durations.")
    
    event_starts = np.sort(np.random.choice(available_start_positions, num_spikes, replace=False))
    
    for start in event_starts:
        # Randomly determine if the spike will be positive or negative
        direction = np.random.choice([-1, 1])
        
        # Randomly determine the amplitude of this spike
        amplitude = np.random.uniform(0.1, max_spike_amplitude) * direction
        
        # Create spike
        spike_shape = amplitude * (1 - np.cos(np.linspace(0, np.pi, spike_duration))) / 2
        viscosity_variation[start:start+spike_duration] += spike_shape
        
        # Create plateau
        plateau_level = spike_shape[-1]
        viscosity_variation[start+spike_duration:start+event_duration] += plateau_level
    
    # Ensure viscosity stays positive
    viscosity_variation = np.maximum(viscosity_variation, 0.1 * base_viscosity)
    
    return viscosity_variation



def evaluate_policy_ae(env, policy, max_episode_steps, deterministic=True, num_ae_features=23):
    # Generate gravity variation for the episode
    gravity_values = generate_gravity_variation(max_episode_steps, variation_type='step_like_pattern')
    wind_values = generate_wind_variation(max_episode_steps)
    viscosity_values = generate_viscosity_variation(max_episode_steps)

    obs = env.reset()
    # obs = (obs - mean) / std_dev
    
    total_reward = 0.
    hidden_state = torch.zeros(1, 1)
    cell_state = torch.zeros(1, num_ae_features * 5).to(DEFAULT_DEVICE)
    
    for step in range(max_episode_steps):
        # Update gravity for the current step

        env.sim.model.opt.viscosity = viscosity_values[step]

        
        action, _, new_cell_state = policy(torchify(obs), hidden_state, cell_state, deterministic)
        action = action.squeeze().detach().cpu().numpy()
        cell_state = new_cell_state
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    
    return total_reward, gravity_values, wind_values, viscosity_values

def eval_policy(env, env_name, policy, max_episode_steps, n_eval_episodes, num_ae_features):
    eval_returns = []
    all_gravity_values = []
    all_wind_values = []
    all_viscosity_values = []

    for _ in range(n_eval_episodes):
        total_reward, gravity_values, wind_values, viscosity_values = evaluate_policy_ae(env=env, policy=policy, max_episode_steps=max_episode_steps, num_ae_features=num_ae_features)
        eval_returns.append(total_reward)
        all_gravity_values.extend(gravity_values)  # Use extend instead of append to concatenate
        all_wind_values.extend(wind_values)
        all_viscosity_values.extend(viscosity_values)

    eval_returns = np.array(eval_returns)
    all_gravity_values = np.array(all_gravity_values)
    all_wind_values = np.array(all_wind_values)
    all_viscosity_values = np.array(all_viscosity_values)

    normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0
    
    results = {
        "return_mean": eval_returns.mean(),
        "return_std": eval_returns.std(),
        "normalized_return_mean": normalized_returns.mean(),
        "normalized_return_std": normalized_returns.std(),
        "gravity_values": all_gravity_values,
        "wind_values": all_wind_values,
        "viscosity_values": all_viscosity_values,
        "max_episode_steps": max_episode_steps,
        "n_eval_episodes": n_eval_episodes
    }

    return results



######################################################
def plot_mse_and_changes(final_policy, results, env_name):
    mse_loss_history = final_policy.get_mse_loss_history()
    gravity_values = results['gravity_values']

    # Ensure the lengths match (use the shorter length if they differ)
    min_length = min(len(mse_loss_history), len(gravity_values))
    mse_loss_history = mse_loss_history[:min_length]
    gravity_values = gravity_values[:min_length]

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle(f'MSE Loss and Gravity Variation for {env_name}', fontsize=16)

    # Plot MSE Loss
    ax1.plot(mse_loss_history)
    ax1.set_title('MSE Loss over Time')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('MSE Loss')

    # Plot Gravity Variation
    ax2.plot(gravity_values)
    ax2.set_title("Gravity Variation Across All Evaluation Episodes")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Gravity")

    # Add vertical lines to separate episodes in the Gravity plot
    for i in range(1, results['n_eval_episodes']):
        ax2.axvline(x=i * results['max_episode_steps'], color='r', linestyle='--', alpha=0.5)

    # Plot combined MSE Loss and Gravity
    color = 'tab:red'
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('MSE Loss', color=color)
    ax3.plot(mse_loss_history, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    ax3_twin = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax3_twin.set_ylabel('Gravity', color=color)
    ax3_twin.plot(gravity_values, color=color)
    ax3_twin.tick_params(axis='y', labelcolor=color)

    ax3.set_title("MSE Loss and Gravity Variation Over Time")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"mse_and_gravity_{env_name}_AE.png")
    plt.show()

######################################################

#$$$$$$$$$$$$$$$$$$$$$$$$$  wind variation $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def plot_mse_and_wind_changes(final_policy, results, env_name):
    mse_loss_history = final_policy.get_mse_loss_history()
    wind_values = results['wind_values']

    # Ensure the lengths match (use the shorter length if they differ)
    min_length = min(len(mse_loss_history), len(wind_values))
    mse_loss_history = mse_loss_history[:min_length]
    wind_values = wind_values[:min_length]

    # Calculate average wind speed
    avg_wind_speed = np.sqrt(wind_values[:, 0]**2 + wind_values[:, 1]**2)

    # Create a figure with four subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 24))
    fig.suptitle(f'MSE Loss and Wind Variation for {env_name}', fontsize=16)

    # Plot MSE Loss
    ax1.plot(mse_loss_history)
    ax1.set_title('MSE Loss over Time')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('MSE Loss')

    # Plot Wind Variation in X direction
    ax2.plot(wind_values[:, 0])
    ax2.set_title("Wind Variation in X Direction")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Wind Speed (X)")

    # Plot Wind Variation in Y direction
    ax3.plot(wind_values[:, 1])
    ax3.set_title("Wind Variation in Y Direction")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Wind Speed (Y)")

    # Add vertical lines to separate episodes in the Wind plots
    for ax in [ax2, ax3]:
        for i in range(1, results['n_eval_episodes']):
            ax.axvline(x=i * results['max_episode_steps'], color='r', linestyle='--', alpha=0.5)

    # Plot combined MSE Loss and Average Wind Speed
    color = 'tab:red'
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('MSE Loss', color=color)
    ax4.plot(mse_loss_history, color=color)
    ax4.tick_params(axis='y', labelcolor=color)

    ax4_twin = ax4.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax4_twin.set_ylabel('Average Wind Speed', color=color)
    ax4_twin.plot(avg_wind_speed, color=color)
    ax4_twin.tick_params(axis='y', labelcolor=color)

    ax4.set_title("MSE Loss and Average Wind Speed Over Time")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"mse_and_wind_{env_name}_AE.png")
    plt.show()

######################################################



def plot_gradients(gradient_lists, gradient_type, action_dim, env_name):
    """
    Plot gradients for a variable number of actions from a list of lists.
    
    Parameters:
    gradient_lists (list of lists): Each inner list contains gradient values,
                                    one for each action.
    gradient_type (str): Type of gradient being plotted.
    action_dim (int): Number of actions (dimensions).
    env_name (str): Name of the environment.
    """
    # Convert the list of lists to a numpy array for easier manipulation
    gradients = np.array(gradient_lists)
    
    # Ensure we have the correct number of gradients per list
    if gradients.shape[1] != action_dim:
        raise ValueError(f"Each inner list must contain exactly {action_dim} gradient values.")
    
    # Calculate the number of rows and columns for subplots
    n_rows = int(np.ceil(np.sqrt(action_dim)))
    n_cols = int(np.ceil(action_dim / n_rows))
    
    # Create subplots, one for each action
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    fig.suptitle(f'Gradients for {action_dim} Actions - {gradient_type}', fontsize=16)
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # Generate a color cycle
    color_cycle = plt.cm.rainbow(np.linspace(0, 1, action_dim))
    
    # Plot each gradient
    for i in range(action_dim):
        ax = axes[i]
        ax.plot(gradients[:, i], color=color_cycle[i])
        ax.set_title(f'Action {i+1}')
        ax.set_xlabel('Gradient List Index')
        ax.set_ylabel('Gradient Value')
        ax.grid(True)
    
    # Remove any unused subplots
    for i in range(action_dim, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{gradient_type}_grads_{env_name}_AE.png")


def plot_mse_loss(mse_loss_lists, action_dim, env_name):
    """
    Plot MSE loss for a variable number of actions from a list of lists.
    
    Parameters:
    mse_loss_lists (list of lists): Each inner list contains MSE loss values,
                                    one for each action dimension.
    action_dim (int): Number of actions (dimensions).
    env_name (str): Name of the environment.
    """
    # Convert the list of lists to a numpy array for easier manipulation
    mse_losses = np.array(mse_loss_lists)
    
    # Ensure we have the correct number of MSE losses per list
    if mse_losses.shape[1] != action_dim:
        raise ValueError(f"Each inner list must contain exactly {action_dim} MSE loss values.")
    
    # Calculate the number of rows and columns for subplots
    n_rows = int(np.ceil(np.sqrt(action_dim)))
    n_cols = int(np.ceil(action_dim / n_rows))
    
    # Create subplots, one for each action dimension
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    fig.suptitle(f'MSE Loss for {action_dim} Action Dimensions - {env_name}', fontsize=16)
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # Generate a color cycle
    color_cycle = plt.cm.rainbow(np.linspace(0, 1, action_dim))
    
    # Plot each MSE loss
    for i in range(action_dim):
        ax = axes[i]
        ax.plot(mse_losses[:, i], color=color_cycle[i])
        ax.set_title(f'Action Dimension {i+1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE Loss')
        ax.grid(True)
    
    # Remove any unused subplots
    for i in range(action_dim, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()
    plt.savefig(f"mse_loss_{env_name}.png")

def plot_property_variation(data, env_name, property_name):
    """
    Plot the variation of a property over the length of an array.

    Parameters:
    data (np.array): NumPy array containing the property values.
    env_name (str): Name of the environment.
    property_name (str): Name of the property being plotted.
    """
    # Ensure the input is a 1D NumPy array
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input data must be a 1D NumPy array.")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data)), data, color='blue')

    # Set title and labels
    plt.title(f'{property_name} Variation - {env_name}', fontsize=16)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel(property_name, fontsize=12)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize the plot
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # # Add a smoothed trend line (optional)
    # window_size = max(len(data) // 20, 5)  # Adjust smoothing window based on data length
    # smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # plt.plot(range(window_size-1, len(data)), smoothed, color='red', linestyle='--', 
    #          label='Smoothed trend')

    # # Add legend
    # plt.legend(fontsize=10)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    # Save the figure
    plt.savefig(f"{property_name.lower().replace(' ', '_')}_{env_name}.png")
    plt.close()


def main(args):
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Environment: {args.env_name}")
    print(f'Observation dim: {obs_dim}, Action dim: {act_dim}')

    mean = np.array(ast.literal_eval(args.mean), dtype=np.float32)
    std_dev = np.array(ast.literal_eval(args.std_dev), dtype=np.float32)


    ##################################### IQL load  ########################################################################
    deterministic_policy = False
    lstm_args = {
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
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=lstm_args['hidden_dim'], n_hidden=lstm_args['n_hidden'])
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=lstm_args['hidden_dim'], n_hidden=lstm_args['n_hidden'])

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=lstm_args['hidden_dim'], n_hidden=lstm_args['n_hidden']),
        vf=ValueFunction(obs_dim, hidden_dim=lstm_args['hidden_dim'], n_hidden=lstm_args['n_hidden']),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=lstm_args['learning_rate']),
        max_steps=lstm_args['n_steps'],
        tau=lstm_args['tau'],
        beta=lstm_args['beta'],
        alpha=lstm_args['alpha'],
        discount=lstm_args['discount']
    )

    ae_model = LSTM_AE(
        input_dim=obs_dim+act_dim,
        encoding_dim=16,
        h_dims=[32],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh()
    )
    # load weights to iql learning model
    iql.load_state_dict(torch.load(args.iql_model_path, weights_only=True))

    # load weights to ae model
    ae_model.load_state_dict(torch.load(args.ae_model_path, weights_only=True))
    # ae_model.eval()

    num_ae_features = obs_dim + act_dim
    final_policy = AEOpexCellStateModel(iql.policy, ae_model, iql.qf, num_ae_features, mean, std_dev, args.beta, args.gamma, act_dim)

    results = eval_policy(env, args.env_name, final_policy, max_episode_steps=1000, n_eval_episodes=100, num_ae_features=num_ae_features)
    grad_a = final_policy.grad_a_history
    mse_loss_history = final_policy.mse_loss_history

    print(f"Mean Return: {results['return_mean']:.2f} ± {results['return_std']:.2f}")
    print(f"Mean Normalized Return: {results['normalized_return_mean']:.2f} ± {results['normalized_return_std']:.2f}")

    # plot_gradients(mse_grad, 'MSE', action_dim=act_dim, env_name=env_name)
    plot_gradients(grad_a, 'Action', action_dim=act_dim, env_name=args.env_name)
    plot_mse_loss(mse_loss_history, action_dim=act_dim, env_name=args.env_name)

    # plot_property_variation(results["wind_values"][:, 0], env_name, 'Viscosity')
    plot_property_variation(results["viscosity_values"], args.env_name, 'Viscosity')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM AE model")
    parser.add_argument("--env_name", type=str, help="Name of the environment")
    parser.add_argument("--iql_model_path", type=str, help="Path to the trained IQL model")
    parser.add_argument("--ae_model_path", type=str,  help="Path to the trained AE model")
    parser.add_argument("--mean", type=str,  help="Mean value for normalization")
    parser.add_argument("--std_dev", type=str, help="Standard deviation for normalization")
    parser.add_argument("--beta", type=float, help="Beta value for the AE model")
    parser.add_argument("--gamma", type=float, help="Gamma value for the AE model")
    args = parser.parse_args()
    main(args)
    