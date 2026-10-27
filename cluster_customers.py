from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def main():
    rfm = pd.read_csv(OUTPUTS / "customer_rfm.csv")

    # Scale the three features so spending does not dominate
    # simply because its numerical values are larger.
    features = rfm[["recency", "frequency", "monetary"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    results = []

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        )

        labels = model.fit_predict(scaled)

        results.append({
            "K": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(scaled, labels),
        })

    comparison = pd.DataFrame(results)

    print("K-MEANS COMPARISON")
    print(comparison.round(4).to_string(index=False))

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUTS / "kmeans_comparison.csv", index=False)

    print("\nSaved: outputs/kmeans_comparison.csv")


if __name__ == "__main__":
    main()