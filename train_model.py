import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# -------- Feature Selection --------
# 'duration' is always ~60s — not useful
# 'eye_contact_time' is duplicate of eye_contact_percentage
df.drop(columns=["duration", "eye_contact_time"], inplace=True)

# -------- Define X and y --------
X = df.drop(columns=["label"])
y = df["label"]

print("Features selected:", list(X.columns))
print("X shape:", X.shape)


# -------- Train/Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80% train, 20% test
    random_state=42,     # same split every time
    stratify=y           # keep 50/50 balance in both splits
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# -------- Scale Features --------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learn scale from train only
X_test_scaled  = scaler.transform(X_test)        # apply same scale to test

print("Scaling done ")


# -------- Train Logistic Regression --------
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)
print("Logistic Regression trained ")

# -------- Train Random Forest --------
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    max_depth=4,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)
print("Random Forest trained ")
# -------- Evaluate Logistic Regression --------
lr_pred = lr_model.predict(X_test_scaled)
print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred, target_names=["Typical", "Atypical"]))

# -------- Evaluate Random Forest --------
rf_pred = rf_model.predict(X_test_scaled)
print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred, target_names=["Typical", "Atypical"]))

# -------- Confusion Matrix (Random Forest) --------
cm = confusion_matrix(y_test, rf_pred)
ConfusionMatrixDisplay(cm, display_labels=["Typical", "Atypical"]).plot()
plt.title("Random Forest - Confusion Matrix")
plt.savefig("dataset/confusion_matrix.png")
print("Confusion matrix saved ")