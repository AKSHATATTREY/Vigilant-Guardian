import pandas as pd

# Load your datasets
neutral_df = pd.read_csv("neutral.txt")
resting_df = pd.read_csv("resting.txt")
holding_df = pd.read_csv("holding.txt")
gripping_df = pd.read_csv("gripping.txt")
punches_df = pd.read_csv("punches.txt")

# Check the shape of each DataFrame
datasets = {
    "Neutral": neutral_df,
    "Resting": resting_df,
    "Holding": holding_df,
    "Gripping": gripping_df,
    "Punches": punches_df,
}

for name, df in datasets.items():
    print(f"{name} shape: {df.shape}")

# Ensure all datasets have the same number of columns
num_columns = [df.shape[1] for df in datasets.values()]
if all(col == num_columns[0] for col in num_columns):
    print("All datasets have the same number of columns.")
else:
    print("Datasets have different number of columns.")
