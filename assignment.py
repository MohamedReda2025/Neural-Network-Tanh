import numpy as np
i1 = 0.05
i2 = 0.10
b1 = 0.5
b2 = 0.7

w1, w2, w3, w4, w5, w6, w7, w8 = np.random.uniform(-0.5, 0.5, 8)

def tanh(x):
    return np.tanh(x)
net_h1 = (w1 * i1) + (w2 * i2) + b1
out_h1 = tanh(net_h1)

net_h2 = (w3 * i1) + (w4 * i2) + b1
out_h2 = tanh(net_h2)
net_o1 = (w5 * out_h1) + (w6 * out_h2) + b2
out_o1 = tanh(net_o1)
net_o2 = (w7 * out_h1) + (w8 * out_h2) + b2
out_o2 = tanh(net_o2)

print("--- Initial Parameters ---")
print(f"Inputs: i1 = {i1}, i2 = {i2}")
print(f"Biases: b1 = {b1}, b2 = {b2}")
print(f"Weights (w1-w8): {[round(w, 4) for w in [w1, w2, w3, w4, w5, w6, w7, w8]]}\n")

print("--- Network Final Outputs ---")
print(f"Output of neuron o1: {out_o1:.4f}")

print(f"Output of neuron o2: {out_o2:.4f}")

