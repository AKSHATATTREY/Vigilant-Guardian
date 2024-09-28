import pandas as pd

# Load datasets
neutral_df = pd.read_csv("neutral.txt")
resting_df = pd.read_csv("resting.txt")
holding_df = pd.read_csv("holding.txt")
gripping_df = pd.read_csv("gripping.txt")
punches_df = pd.read_csv("punches.txt")

# Display initial shapes
print(f"Neutral shape: {neutral_df.shape}")
print(f"Resting shape: {resting_df.shape}")
print(f"Holding shape: {holding_df.shape}")
print(f"Gripping shape: {gripping_df.shape}")
print(f"Punches shape: {punches_df.shape}")

# Define the target number of columns (from one of the other datasets)
target_columns = neutral_df.shape[1]  # You can choose any dataset

# Align punches_df to match the target number of columns
if punches_df.shape[1] > target_columns:
    punches_df = punches_df.iloc[:, :target_columns]  # Trim to match
elif punches_df.shape[1] < target_columns:
    # Add columns with NaN values if fewer columns
    punches_df = pd.concat([punches_df, pd.DataFrame(columns=range(punches_df.shape[1], target_columns))], axis=1)

# Save the modified punches_df back to the file
punches_df.to_csv("punches.txt", index=False)

# Display new shape
print(f"Updated Punches shape: {punches_df.shape}")
