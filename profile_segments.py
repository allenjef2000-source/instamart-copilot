from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def main():
    rfm = pd.read_csv(OUTPUTS / "customer_rfm.csv")

    features = rfm[["recency", "frequency", "monetary"]]
    scaled = StandardScaler().fit_transform(features)

    model = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=42,
    )

    rfm["segment_id"] = model.fit_predict(scaled)

    profile = rfm.groupby("segment_id").agg(
        customers=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )

    # Premium means the segment with the highest average spending.
    premium_id = profile["mean_monetary"].idxmax()
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)
    profile["is_premium"] = profile.index == premium_id

    print("SEGMENT PROFILES")
    print(profile.round(2).to_string())

    print(f"\nPremium segment ID: {premium_id}")
    print(f"Premium customers: {int(rfm['is_premium'].sum())}")

    # PCA summarises the three scaled RFM features in two components.
    pca = PCA(n_components=2)
    pca.fit(scaled)

    ratios = pca.explained_variance_ratio_

    print("\nPCA RESULTS")
    print(f"Component 1 explained variance: {ratios[0]:.4f}")
    print(f"Component 2 explained variance: {ratios[1]:.4f}")
    print(f"Total explained variance: {ratios.sum():.4f}")

    assert len(rfm) == 1175
    assert profile["customers"].sum() == 1175
    assert profile["is_premium"].sum() == 1
    assert abs(ratios[0] - 0.634) <= 0.02
    assert abs(ratios[1] - 0.218) <= 0.02

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    profile.to_csv(OUTPUTS / "segment_profiles.csv")
    rfm.to_csv(OUTPUTS / "customer_clusters.csv", index=False)

    segment_names = {
        0: "Occasional Customers",
        1: "Premium",
        2: "Frequent Customers",
        3: "Inactive Customers",
    }

    actions = {
        0: "Send relevant product recommendations",
        1: "Offer personalised service and retention benefits",
        2: "Reward loyalty and encourage repeat purchases",
        3: "Try a targeted reactivation campaign",
    }

    rfm["segment_name"] = rfm["segment_id"].map(segment_names)

    # Premium must always identify the highest-spending segment.
    assert segment_names[premium_id] == "Premium"

    columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]

    assert rfm["segment_name"].notna().all()

    rfm[columns].to_csv(
        OUTPUTS / "customer_segments.csv",
        index=False,
    )

    profile["segment_name"] = profile.index.map(segment_names)
    profile["recommended_action"] = profile.index.map(actions)
    profile.to_csv(OUTPUTS / "segment_profiles.csv")

    print("\nFINAL NAMED SEGMENTS")
    print(
        profile[
            ["segment_name", "customers", "recommended_action"]
        ].to_string()
    )
    print("\nSaved: outputs/customer_segments.csv")
    print("\nSaved: outputs/segment_profiles.csv")
    print("Saved: outputs/customer_clusters.csv")
    print("SEGMENT PROFILE CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()