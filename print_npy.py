import numpy as np

# Load the .npy file
file_path = 'npy/cairovirtual-x.npy'
data = np.load(file_path, allow_pickle=True)

# Print the content of the array
print("Contents of the .npy file:")
#for i in data:
    #print(i)

print("X_train_input:")
print(type(data), data.dtype, data.shape)

# print("\nY[train]:")
# print(type(Y[train]), Y[train].dtype if isinstance(Y[train], np.ndarray) else "Not a NumPy array")


# If the array is too large, summarize it
print("\nSummary:")
print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
