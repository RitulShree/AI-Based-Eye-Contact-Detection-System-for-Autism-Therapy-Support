import pandas as pd

# -------- Load Both CSVs --------
cols = [
    "session_id", "session_type", "duration", "blink_rate",
    "avg_closure_duration", "avg_IBI", "IBI_std", "long_closures",
    "total_fixations", "avg_fixation_duration", "avg_movement",
    "gaze_variance", "longest_no_blink", "eye_contact_time",
    "eye_contact_percentage"
]

df_yours = pd.read_csv("dataset/gaze_dataset.csv", header=None, names=cols)
df_hers  = pd.read_csv("dataset/gaze_dataset_ritul.csv")

# -------- Merge --------
df = pd.concat([df_yours, df_hers], ignore_index=True)

# -------- Drop session_id (not a feature) --------
df.drop(columns=["session_id"], inplace=True)

# -------- Convert label to 0/1 --------
df["label"] = df["session_type"].map({"typical": 0, "atypical": 1})
df.drop(columns=["session_type"], inplace=True)

# -------- Save merged dataset --------
df.to_csv("dataset/merged_dataset.csv", index=False)

# -------- Verify --------
print("Shape:", df.shape)
print("Label balance:\n", df["label"].value_counts())
print("Missing values:\n", df.isnull().sum())
print("First 3 rows:\n", df.head(3))