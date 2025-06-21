import pandas as pd

# Load your labeled CSV (long-form: label,x,y,image,width,height)
df = pd.read_csv("labels_my-project-name_2025-06-02-12-18-10.csv", header=None)
df.columns = ["label", "x", "y", "image", "width", "height"]
df["v"] = 1  # Visibility flag: 1 (visible)

# Pivot to wide format
pivot_x = df.pivot_table(index="image", columns="label", values="x")
pivot_y = df.pivot_table(index="image", columns="label", values="y")
pivot_v = df.pivot_table(index="image", columns="label", values="v")

# Reconstruct a flat table with x/y/v grouped per keypoint
all_labels = sorted(set(df["label"]))  # sort for consistency
data = {"image": pivot_x.index}
for label in all_labels:
    data[f"{label}_x"] = pivot_x[label] if label in pivot_x else -1
    data[f"{label}_y"] = pivot_y[label] if label in pivot_y else -1
    data[f"{label}_v"] = pivot_v[label] if label in pivot_v else 0

# Convert to DataFrame
final_df = pd.DataFrame(data)

# Fill any remaining NaNs with -1 or 0
final_df.fillna({col: -1 for col in final_df.columns if "_x" in col or "_y" in col}, inplace=True)
final_df.fillna({col: 0 for col in final_df.columns if "_v" in col}, inplace=True)

# Save to CSV
final_df.to_csv("standing_quad_pose.csv", index=False)

print("✅ CSV saved with x/y/v columns grouped per keypoint.")
