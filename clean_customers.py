from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(
    ROOT / "data" / "raw" / "customers.csv",
    keep_default_na=False,
)

# 1. Standardise city names.
original_city = customers["city"].copy()
customers["city"] = customers["city"].str.strip().str.title()

city_changes = original_city.ne(customers["city"]).sum()

# 2. Standardise gender values.
original_gender = customers["gender"].copy()

gender_map = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
}

customers["gender"] = (
    customers["gender"]
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("unknown")
)

gender_changes = original_gender.ne(customers["gender"]).sum()

# 3. Convert ages to numbers and mark impossible ages as missing.
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")

invalid_ages = (
    customers["age"].notna()
    & ~customers["age"].between(10, 100)
)

invalid_age_count = invalid_ages.sum()
customers.loc[invalid_ages, "age"] = float("nan")

# Fill missing ages using the median age of each customer's city.
ages_to_fill = customers["age"].isna().sum()

city_median_age = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_age)

# 4. Convert signup dates.
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    format="%Y-%m-%d",
    errors="raise",
)

# 5. Validate the cleaned customer data.
expected_cities = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184,
}

expected_genders = {
    "female": 610,
    "male": 560,
    "unknown": 30,
}

assert len(customers) == 1200
assert customers["customer_id"].is_unique
assert customers["city"].value_counts().to_dict() == expected_cities
assert customers["gender"].value_counts().to_dict() == expected_genders
assert customers["age"].isna().sum() == 0
assert customers["age"].between(18, 60).all()
assert customers["signup_date"].notna().all()

# 6. Save a new cleaned file. The raw file stays unchanged.
output_path = CLEAN / "customers.csv"

customers.to_csv(
    output_path,
    index=False,
    date_format="%Y-%m-%d",
)

print("CUSTOMER CLEANING SUMMARY")
print(f"City values changed: {city_changes}")
print(f"Gender values changed: {gender_changes}")
print(f"Invalid ages marked missing: {invalid_age_count}")
print(f"Missing ages filled: {ages_to_fill}")

print("\nCustomers per city:")
print(customers["city"].value_counts().to_string())

print("\nCustomers per gender:")
print(customers["gender"].value_counts().to_string())

print(f"\nMissing ages remaining: {customers['age'].isna().sum()}")
print(f"Saved to: {output_path}")
print("CUSTOMER CHECKPOINTS PASSED")
