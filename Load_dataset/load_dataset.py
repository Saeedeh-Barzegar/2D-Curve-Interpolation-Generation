import os
import numpy as np

def load_curves(data_dir, input_size):
    training_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            curve = np.load(os.path.join(data_dir, file))
            training_data.append(curve)
    X = np.array(training_data).reshape([-1, input_size, 2])
    print(f"Loaded {len(X)} samples.")
    return X
