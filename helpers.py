import pandas as pd

# Path to the original CSV
input_csv = "resume_data.csv"

# Path for the new CSV
output_csv = "top_5_applications.csv"

# The row indexes you want to keep
indexes_to_keep = [5581, 1368, 3536, 5213, 893]

# Load the CSV
df = pd.read_csv(input_csv)

# Select only the specified indexes
filtered_df = df.iloc[indexes_to_keep]

# Save to a new CSV
filtered_df.to_csv(output_csv, index=False)

print(f"New CSV saved as: {output_csv}")
