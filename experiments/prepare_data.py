"""Build the three benchmark datasets in a common, public-domain discretized form.

Every column becomes categorical over a domain that is treated as public knowledge
(codebook ranges), as is standard in DP synthetic-data evaluation (e.g. the NIST
challenge). Numeric columns with a large public range are cut into equal-width bins
over that public range, so no data-dependent statistic leaks through preprocessing.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
N_BINS = 16


def _bin_numeric(series, lo, hi, n_bins=N_BINS):
    edges = np.linspace(lo, hi, n_bins + 1)
    clipped = series.clip(lo, hi)
    idx = np.clip(np.searchsorted(edges, clipped, side="right") - 1, 0, n_bins - 1)
    return idx.astype(int).astype(str), edges.tolist()


def build(df, label, numeric_bounds, name):
    """numeric_bounds: {col: (lo, hi)} for columns to bin; every other column is categorical."""
    out = pd.DataFrame(index=df.index)
    schema = {"name": name, "label": label, "columns": {}}
    for c in df.columns:
        if c in numeric_bounds:
            lo, hi = numeric_bounds[c]
            out[c], edges = _bin_numeric(df[c].astype(float), lo, hi)
            schema["columns"][c] = {"type": "binned", "edges": edges, "domain": [str(i) for i in range(N_BINS)]}
        else:
            vals = df[c].astype(str)
            out[c] = vals
            schema["columns"][c] = {"type": "categorical", "domain": sorted(vals.unique().tolist())}
    out.to_csv(os.path.join(DATA, f"{name}.csv"), index=False)
    with open(os.path.join(DATA, f"{name}.schema.json"), "w") as f:
        json.dump(schema, f, indent=1)
    print(name, out.shape, "label dist:", out[label].value_counts(normalize=True).round(3).to_dict())


def brfss():
    df = pd.read_csv(os.path.join(HERE, "..", "input_data", "diabetes_012_health_indicators_BRFSS2021.csv"))
    df["Diabetes_012"] = df["Diabetes_012"].astype(int)
    # BRFSS 2021 codebook: BMI recorded 12-99; every other column already has a small coded domain.
    for c in df.columns:
        if c != "BMI":
            df[c] = df[c].astype(int)
    build(df, "Diabetes_012", {"BMI": (12.0, 99.0)}, "brfss")


def adult():
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    a = pd.read_csv(os.path.join(DATA, "adult", "adult.data"), names=cols, skipinitialspace=True)
    b = pd.read_csv(os.path.join(DATA, "adult", "adult.test"), names=cols, skipinitialspace=True, skiprows=1)
    df = pd.concat([a, b], ignore_index=True)
    df["income"] = df["income"].str.replace(".", "", regex=False).map({"<=50K": 0, ">50K": 1}).astype(int)
    # fnlwgt is a survey weight and education_num duplicates education: both are dropped by convention.
    df = df.drop(columns=["fnlwgt", "education_num"])
    bounds = {"age": (17, 90), "capital_gain": (0, 99999), "capital_loss": (0, 4356), "hours_per_week": (1, 99)}
    build(df, "income", bounds, "adult")


def _icd9_group(code):
    """Strack et al. (2014) grouping of primary ICD-9 codes for this dataset."""
    if not isinstance(code, str) or code == "?":
        return "Missing"
    if code.startswith(("V", "E")):
        return "Other"
    v = float(code)
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def diabetes130():
    df = pd.read_csv(os.path.join(DATA, "diabetes130", "diabetic_data.csv"), low_memory=False)
    # DP protects individuals, so keep one row per patient (their first encounter).
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    # Drop identifiers, discharges to hospice/death (cannot be readmitted), and columns >40% missing.
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
    df["readmitted"] = (df["readmitted"] == "<30").astype(int)
    for d in ["diag_1", "diag_2", "diag_3"]:
        df[d] = df[d].map(_icd9_group)
    keep = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id", "admission_source_id",
            "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
            "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
            "max_glu_serum", "A1Cresult", "metformin", "glipizide", "glyburide", "pioglitazone",
            "rosiglitazone", "insulin", "change", "diabetesMed", "readmitted"]
    df = df[keep]
    bounds = {"num_lab_procedures": (1, 132), "num_medications": (1, 81), "number_outpatient": (0, 42),
              "number_emergency": (0, 76), "number_inpatient": (0, 21)}
    build(df, "readmitted", bounds, "diabetes130")


if __name__ == "__main__":
    brfss()
    adult()
    diabetes130()
