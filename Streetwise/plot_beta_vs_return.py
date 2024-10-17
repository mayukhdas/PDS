import numpy as np
import matplotlib.pyplot as plt

# Data
eval_types = ['Only Action', '0.1', '0.01', '0.001', '0.9', '0.5', '0.25', '0.375', '0.3125']
means = [47.2, 51.92, 48.59, 47.41, 18.25, 36.62, 49.87, 44.18, 46.71]

# Convert beta values to float, excluding 'Only Action'
betas = [float(b) for b in eval_types[1:]]

# Plotting
plt.figure(figsize=(16, 10))

# Scatter plot of actual data points
scatter = plt.scatter(betas, means[1:], color='blue', s=100, label='Mean Return')

# Extrapolation before beta=0.001
extrapolation_beta = [0.0001, 0.001]  # Add a point before 0.001
extrapolation_mean = [means[0], means[3]]  # Extrapolate to 'Only Action' mean
plt.plot(extrapolation_beta, extrapolation_mean, color='blue', linestyle='--', label='Extrapolation')

# Only Action line
plt.axhline(y=means[0], color='r', linestyle='--', label='Only Action')

# Set x-axis to log scale
plt.xscale('log')
plt.xlabel('Beta', fontsize=12)
plt.ylabel('Mean Return', fontsize=12)
plt.title('Mean Return vs Beta - Without Noise', fontsize=16)
plt.grid(True, which="both", ls="-", alpha=0.2)

# Highlight the range where mean is higher than 'Only Action'
plt.axvspan(0.001, 0.3125, alpha=0.2, color='green', label='Better than Only Action')

# Highlight the extrapolated range where mean is expected to be worse
plt.axvspan(1e-4, 0.001, alpha=0.2, color='orange', label='Worse than Only Action (Extrapolated)')

# Highlight the observed range where mean is worse
worse_mean_mask = np.array(means[1:]) < means[0]
worse_betas = np.array(betas)[worse_mean_mask]
if len(worse_betas) > 0:
    plt.axvspan(min(worse_betas), max(betas), alpha=0.2, color='red', label='Worse than Only Action (Observed)')

# Annotate each point with its beta value and mean return
for beta, mean in zip(betas, means[1:]):
    plt.annotate(f'β: {beta}\nMean: {mean:.2f}', 
                 (beta, mean),
                 textcoords="offset points", 
                 xytext=(0,15), 
                 ha='center',
                 va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 fontsize=9)

# Annotate 'Only Action' point
plt.annotate(f'Only Action\nMean: {means[0]}', 
             (min(betas), means[0]),
             textcoords="offset points", 
             xytext=(-10, -15), 
             ha='right',
             va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
             fontsize=9)

plt.legend(fontsize=10, loc='lower left')
plt.tight_layout()
plt.show()

plt.savefig('beta_vs_return_without_noise.png')

# Print the range of beta where performance is better
print(f"The range of beta where mean return is better: 0.001 to 0.3125")

# Print the range of beta where performance is worse
print(f"The range of beta where mean return is worse:")
print(f"  - Below 0.001 (extrapolated)")
print(f"  - Above {min(worse_betas)} (observed)")

# Print the best beta
best_beta = betas[np.argmax(means[1:])]
print(f"The beta with the highest mean return: {best_beta}")