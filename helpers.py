import pandas as pd

dataset_df = pd.read_csv("dataset.csv")
main_df = pd.read_csv("resume_data.csv")

print(main_df["skills_required"])
print(main_df["related_skils_in_job"])
