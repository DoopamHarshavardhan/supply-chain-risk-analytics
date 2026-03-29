import pickle
import pandas as pd

with open("ml_outputs/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("ml_outputs/features.pkl", "rb") as f:
    features = pickle.load(f)

importance = model.feature_importances_
df = pd.DataFrame({"Feature": features, "Importance": importance})
df = df.sort_values("Importance", ascending=False)
df.to_csv("ml_outputs/feature_importance.csv", index=False)
print("Done! feature_importance.csv saved.")
print(df.head(10))


import pickle

with open("ml_outputs/features.pkl", "rb") as f:
    features = pickle.load(f)

with open("ml_outputs/columns.txt", "w") as f:
    for col in features:
        f.write(col + "\n")

print("Done! columns.txt saved.")
print(f"Total columns: {len(features)}")