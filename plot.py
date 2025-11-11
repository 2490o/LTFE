import matplotlib.pyplot as plt
import numpy as np

# Hyperparameter values (τ)
tau_values = np.linspace(0, 0.1, 11)

# mIoU values
miou_values_tau = [37.5, 38.0, 39.0, 40.2, 41.0, 40.7, 39.8, 39.3, 39.0, 38.4, 37.8]

# Plotting the line chart with star markers
plt.plot(tau_values, miou_values_tau, 'b--', label="Average mIoU", marker='*', markersize=8)

# Adding labels and title
plt.xlabel("Hyperparameter τ")
plt.ylabel("mIoU (%)")
plt.title("(b) Hyperparameter τ")

# Show grid lines
plt.grid(True)

# Add legend
plt.legend()

# Display the plot
plt.show()
