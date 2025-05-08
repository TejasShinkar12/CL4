import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the triangular membership function
center = 30
left_boundary = 20
right_boundary = 40

# Generate x values for plotting
x = np.linspace(0, 60, 100)

# Calculate the membership function values using triangular shape
membership = np.maximum(0, np.minimum((x - left_boundary) / (center - left_boundary),
                                      (right_boundary - x) / (right_boundary - center)))

# Plot the membership function
plt.figure(figsize=(8, 6))
plt.plot(x, membership, 'b', linewidth=2)
plt.title('Triangular Membership Function for Linguistic Term "Low"', fontsize=16)
plt.xlabel('Car Speed (km/h)', fontsize=14)
plt.ylabel('Degree of Membership', fontsize=14)
plt.grid(True)
plt.show()
