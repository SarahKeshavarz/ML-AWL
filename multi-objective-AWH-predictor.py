# %%
# === PIP INSTALLS === installing the required tools
!pip install lightgbm scikit-learn pandas numpy joblib matplotlib seaborn

# %%
# === IMPORTS === importing the required tools
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
import joblib  


# %%
# === LOAD & PREPROCESS CLASSIFICATION DATA ===
df_cls = pd.read_csv('Wstability-data.csv')
df_cls['MOF_name'] = df_cls['MOF_name'].astype(str).str.strip()

# Convert only numeric columns (exclude MOF_name and water_label)
non_numeric_cols = ['MOF_name', 'water_label']
numeric_cols = [col for col in df_cls.columns if col not in non_numeric_cols]

df_cls[numeric_cols] = df_cls[numeric_cols].apply(pd.to_numeric, errors='coerce')
df_cls.replace(['#DIV/0!', '#NAME?', 'NaN', 'nan'], np.nan, inplace=True)

# Fill NaNs only in numeric columns with their mean
df_cls[numeric_cols] = df_cls[numeric_cols].fillna(df_cls[numeric_cols].mean())

# === SET FEATURES & TARGET ===
rfa_features = [
    'mc-I-3-all', 'D_lc-T-1-all', 'mc-Z-0-all', 'func-I-1-all', 'f-lig-I-3',
    'func-I-0-all', 'D_lc-S-3-all', 'f-lig-I-0', 'KHW', 'D_mc-S-1-all'
]

y_cls = df_cls["water_label"].astype(int).replace({1: 0, 2: 0, 3: 1, 4: 1})
if y_cls.min() == 1:
    y_cls -= 1

# Handle missing classification features dynamically
available_cls_features = [f for f in rfa_features if f in df_cls.columns]
missing_cls_features = [f for f in rfa_features if f not in df_cls.columns]
if missing_cls_features:
    print(f"Warning: Missing features {missing_cls_features} will be skipped.")

X_cls = df_cls[available_cls_features].copy()
X_cls.replace([np.inf, -np.inf], np.nan, inplace=True)
X_cls.fillna(X_cls.mean(), inplace=True)


# === TRAIN CLASSIFIER ===
kf = KFold(n_splits=10, shuffle=True, random_state=42)
model_cls = LGBMClassifier(random_state=42, n_jobs=-1)  # default params
scaler_cls = StandardScaler()

for train_idx, test_idx in kf.split(X_cls):
    X_train = scaler_cls.fit_transform(X_cls.iloc[train_idx])
    y_train = y_cls.iloc[train_idx]
    model_cls.fit(X_train, y_train)

# Save trained classifier
joblib.dump(model_cls, "classifier_water_stability.pkl")

# === LOAD & PREPROCESS REGRESSION DATA ===
df_reg = pd.read_csv('WUSdata-mod.csv')
df_reg['MOF_name'] = df_reg['MOF_name'].astype(str).str.strip()
df_reg.set_index('MOF_name', inplace=True)

# EXACT preprocessing order for regression
df_reg = df_reg.apply(pd.to_numeric, errors='coerce')
df_reg.replace(['#DIV/0!', '#NAME?', 'NaN', 'nan'], np.nan, inplace=True)
df_reg.fillna(df_reg.mean(), inplace=True)

# === IDENTIFY KNOWN AND UNKNOWN STABILITY MOFs ===
mofs_cls_set = set(df_cls['MOF_name'])
mofs_reg_set = set(df_reg.index)

known_mofs = list(mofs_reg_set & mofs_cls_set)
unknown_mofs = list(mofs_reg_set - mofs_cls_set)

df_known = df_reg.loc[known_mofs].copy()
df_unknown = df_reg.loc[unknown_mofs].copy()

# === PREDICT STABILITY ===
available_reg_features = [f for f in rfa_features if f in df_known.columns]
missing_reg_features = [f for f in rfa_features if f not in df_known.columns]
if missing_reg_features:
    print(f"Warning: Missing features {missing_reg_features} will be skipped.")

X_known = df_known[available_reg_features].copy()
X_known = scaler_cls.transform(X_known)
df_known['water_stability_prob'] = model_cls.predict_proba(X_known)[:, 1]

X_unknown = df_unknown[available_reg_features].copy()
X_unknown = scaler_cls.transform(X_unknown)
df_unknown['water_stability_prob'] = model_cls.predict_proba(X_unknown)[:, 1]

df_full = pd.concat([df_known, df_unknown])

# === NUMERIC TARGET COLUMNS ===
for target in ['Wselectivity', 'Wup20cm3', 'Wup80cm3']:
    df_full[target] = pd.to_numeric(df_full[target], errors='coerce')

# === DESCRIPTORS FOR REGRESSION ===
descriptors_indices = [1, 195]
headers = list(df_full.columns)
descriptors = headers[min(descriptors_indices):max(descriptors_indices) + 1]

# TRAIN AND SAVE REGRESSORS
for target in ['Wselectivity', 'Wup20cm3', 'Wup80cm3']:
    y = df_full[target]
    X = df_full[[f for f in descriptors if f in df_full.columns]]
    model = LGBMRegressor(
        num_leaves=60,
        n_estimators=400,
        max_depth=-1,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    joblib.dump(model, f"regressor_{target}.pkl")

# === APPLY STABILITY FILTER ===
stable = df_full[df_full['water_stability_prob'] >= 0.9].dropna(
    subset=['Wselectivity', 'Wup20cm3', 'Wup80cm3']
)

# === STATISTICAL FILTERING ===
means = stable[['Wup20cm3', 'Wup80cm3']].mean()
stds = stable[['Wup20cm3', 'Wup80cm3']].std()

filtered = stable[
    (stable['Wselectivity'] >= 1000) &
    (stable['Wup20cm3'] >= means['Wup20cm3'] + stds['Wup20cm3']) &
    (stable['Wup80cm3'] >= means['Wup80cm3'] + stds['Wup80cm3'])
]

# === OUTPUT SORTED TOP MOFs ===
top_mofs = filtered.sort_values(by='Wup20cm3', ascending=False)
print("=== Top MOFs for AWH (Rule-based filtered) ===")
print(top_mofs.head(100))

final_features = [
    'D_lc-S-3-all', 'D_lc-T-1-all', 'D_mc-S-1-all', 'Density-mcs',
    'Df', 'Di', 'KHW', 'LCD-mcs', 'PLD-mcs', 'VF-mcs', 'VPOV',
    'f-lig-I-0', 'f-lig-I-3', 'func-I-0-all', 'func-I-1-all',
    'mc-I-3-all', 'mc-Z-0-all', 'mc-chi-2-all', 'mc-chi-3-all',
    'unit_cell_volume', 'Wselectivity', 'Wup20cm3', 'Wup80cm3',
    'water_stability_prob'
]

available_features = [f for f in final_features if f in top_mofs.columns]
top_mofs[available_features].head(100).to_csv('top-100-MOFs-with-features.csv', index_label='MOF_name')


# %%
# === ASK USER FOR INPUT FILE ===
input_file = input("Enter the input CSV file name (e.g. test-MOF.csv): ").strip()
df_input = pd.read_csv(input_file)
df_input['MOF_name'] = df_input['MOF_name'].astype(str).str.strip()
df_input.set_index('MOF_name', inplace=True)

# EXACT preprocessing order
df_input = df_input.apply(pd.to_numeric, errors='coerce')
df_input.replace(['#DIV/0!', '#NAME?', 'NaN', 'nan'], np.nan, inplace=True)
df_input.fillna(df_input.mean(), inplace=True)

# === CLASSIFICATION: PREDICT WATER STABILITY ===
available_cls_feats = [f for f in rfa_features if f in df_input.columns]
missing_cls_feats = [f for f in rfa_features if f not in df_input.columns]
if missing_cls_feats:
    print(f"Warning: Missing classification features {missing_cls_feats} will be skipped.")

X_input_cls = df_input[available_cls_feats].copy()
X_input_cls = pd.DataFrame(
    SimpleImputer(strategy="mean").fit_transform(X_input_cls),
    index=df_input.index,
    columns=available_cls_feats
)

for feat in rfa_features:
    if feat not in X_input_cls.columns:
        X_input_cls[feat] = 0

X_input_cls = X_input_cls[rfa_features]
X_input_cls = scaler_cls.transform(X_input_cls)

model_cls = joblib.load("classifier_water_stability.pkl")
df_input['water_stability_prob'] = model_cls.predict_proba(X_input_cls)[:, 1]

# === REGRESSION: PREDICT TARGETS IF MISSING ===
targets = ['Wselectivity', 'Wup20cm3', 'Wup80cm3']
missing_targets = [t for t in targets if t not in df_input.columns]
print(f"Missing target values: {missing_targets}" if missing_targets else "✅ All target values provided.")

# descriptors from full dataset
headers = list(df_input.columns)
descriptors = headers[min(descriptors_indices):max(descriptors_indices) + 1]

for target in targets:
    model = joblib.load(f"regressor_{target}.pkl")
    if target not in df_input.columns:
        df_input[target] = np.nan

    X_pred_reg = df_input[[f for f in descriptors if f in df_input.columns]].copy()
    X_pred_reg = X_pred_reg.fillna(X_pred_reg.mean())

    missing_mask = df_input[target].isnull()
    df_input.loc[missing_mask, target] = model.predict(X_pred_reg[missing_mask])

# === EVALUATE RULE-BASED SUITABILITY ===
means = df_full[['Wup20cm3', 'Wup80cm3']].mean()
stds = df_full[['Wup20cm3', 'Wup80cm3']].std()

suitable = df_input[
    (df_input['water_stability_prob'] >= 0.75) &
    (df_input['Wselectivity'] >= 1000) &
    (df_input['Wup20cm3'] >= means['Wup20cm3'] + stds['Wup20cm3']) &
    (df_input['Wup80cm3'] >= means['Wup80cm3'] + stds['Wup80cm3'])
]

# === REPORT RESULTS ===
print("\n=== Input MOF Analysis ===")
print(df_input[['water_stability_prob', 'Wup20cm3', 'Wup80cm3', 'Wselectivity']])

print("\n=== Suitable MOFs for Atmospheric Water Harvesting ===")
if not suitable.empty:
    for mof in suitable.index:
        print(f"{mof}: Suitable for AWH")
else:
    print(" No suitable MOFs found based on the criteria.")



