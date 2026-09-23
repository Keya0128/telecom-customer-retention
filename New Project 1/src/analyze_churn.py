from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  
DATA = ROOT / "data" / "telecom_churn_synthetic.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# 2. Feature Mapping
df["churn_flag"] = df["churn"].map({"Yes": 1, "No": 0})
df["tenure_band"] = pd.cut(
    df["tenure_months"], 
    bins=[0, 6, 12, 24, 48, 72],
    labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    include_lowest=True,
)

# 3. Defensive Data Quality Assertions - CORRECTED FOR TELECOM
print("\n Running Data Quality Integrity Check...")

# 1. Check for Duplicate Customers
duplicate_count = df.duplicated(subset="customer_id").sum()
if duplicate_count > 0:
    print(f" -> WARNING: Found {duplicate_count} duplicate customer records! Cleaning them now...")
    df = df.drop_duplicates(subset="customer_id", keep="first")
else:
    print(" -> PASS: All customer_id records are uniquely verified.")

# 2. Check for Negative Charges Errors
negative_charges_count = (df["monthly_charges"] < 0).sum()
if negative_charges_count > 0:
    print(f" -> WARNING: Found {negative_charges_count} rows with negative monthly_charges! Removing errors...")
    df = df[df["monthly_charges"] >= 0]
else:
    print(" -> PASS: Monthly charges contain zero negative anomalies.")

# 3. Check for Missing Blank Churn Targets
missing_churn_count = df["churn"].isna().sum()
if missing_churn_count > 0:
    print(f" -> WARNING: Found {missing_churn_count} rows with blank Churn fields! Removing rows...")
    df = df.dropna(subset=["churn"])
else:
    print(" -> PASS: Target Churn field data is 100% complete.")

# 4. Check for Infinite Errors in total_charges
# First convert to numeric because your raw data has blank strings
df["total_charges"] = pd.to_numeric(df["total_charges"], errors='coerce')
has_infinite_values = np.isinf(df["total_charges"]).any() if df["total_charges"].notna().any() else False

if has_infinite_values:
    inf_count = np.isinf(df["total_charges"]).sum()
    print(f" -> WARNING: Found {inf_count} rows with Infinite numbers! Cleaning them now...")
    df["total_charges"] = df["total_charges"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["total_charges"])
else:
    print(" -> PASS: Total charges contains 100% stable, finite numbers.")

print("Data Integrity verification finalized successfully.\n")

# 4. Sorting Key for Power BI
band_mapping = {"0-6": 1, "7-12": 2, "13-24": 3, "25-48": 4, "49-72": 5}
df["tenure_sort_index"] = df["tenure_band"].map(band_mapping)

# 5. Export
df.to_csv(OUT / "telecom_churn_clean.csv", index=False)
print("SUCCESS: Clean file stored in outputs/telecom_churn_clean.csv")