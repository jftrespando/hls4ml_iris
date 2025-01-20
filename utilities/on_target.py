# This script runs on the target board
from axi_stream_driver import NeuralNetworkOverlay
import numpy as np

print(f"Loading test data...")
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print(f"Loading bitfile...")
nn = NeuralNetworkOverlay('hls4ml_nn.bit', X_test.shape, y_test.shape)

print(f"Predicting...")
y_hw, latency, throughput = nn.predict(X_test, profile=True)

print(f"Saving results...")
np.save('y_hw.npy', y_hw)