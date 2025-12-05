import pandas as pd

# Load CSV
df = pd.read_csv("Dataset/reddit_data.csv")

# Convert TRUE/FALSE strings to boolean if needed
df = df.replace({"TRUE": True, "FALSE": False})

# Count TRUEs for each abuse type
physical_true = df["Physical Abuse"].sum()
emotional_true = df["Emotional Abuse"].sum()
sexual_true = df["Sexual Abuse"].sum()

# Count Tag == FALSE
tag_false = (df["Tag"] == False).sum()

# Print results
print("Counts:")
print(f"Physical Abuse = TRUE : {physical_true}")
print(f"Emotional Abuse = TRUE: {emotional_true}")
print(f"Sexual Abuse = TRUE  : {sexual_true}")
print(f"Tag = FALSE          : {tag_false}")