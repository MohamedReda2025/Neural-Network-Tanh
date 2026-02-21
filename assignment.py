import numpy as np

# 1. Inputs (From the previous network in the lecture)
i1 = 0.05
i2 = 0.10

# 2. Biases (Given in the exercise)
b1 = 0.5
b2 = 0.7

# 3. Weights: Random from interval [-0.5, 0.5]
# We need 8 weights (w1 to w8) for the network architecture
# Using a seed so the random values change every time you run, 
# but you can remove the seed if you want completely random results each run.
w1, w2, w3, w4, w5, w6, w7, w8 = np.random.uniform(-0.5, 0.5, 8)

# 4. Tanh Activation Function
def tanh(x):
    return np.tanh(x)

# ---------------------------------------------------------
# FORWARD PASS (Following the lecture's step-by-step method)
# ---------------------------------------------------------

# Step 1: Hidden Layer calculations
# Calculating net and out for h1
net_h1 = (w1 * i1) + (w2 * i2) + b1
out_h1 = tanh(net_h1)

# Calculating net and out for h2
net_h2 = (w3 * i1) + (w4 * i2) + b1
out_h2 = tanh(net_h2)


# Step 2: Output Layer calculations
# Calculating net and out for o1
net_o1 = (w5 * out_h1) + (w6 * out_h2) + b2
out_o1 = tanh(net_o1)

# Calculating net and out for o2
net_o2 = (w7 * out_h1) + (w8 * out_h2) + b2
out_o2 = tanh(net_o2)


# ---------------------------------------------------------
# 5. Print the output of the network
# ---------------------------------------------------------
print("--- Initial Parameters ---")
print(f"Inputs: i1 = {i1}, i2 = {i2}")
print(f"Biases: b1 = {b1}, b2 = {b2}")
print(f"Weights (w1-w8): {[round(w, 4) for w in [w1, w2, w3, w4, w5, w6, w7, w8]]}\n")

print("--- Network Final Outputs ---")
print(f"Output of neuron o1: {out_o1:.4f}")
print(f"Output of neuron o2: {out_o2:.4f}")