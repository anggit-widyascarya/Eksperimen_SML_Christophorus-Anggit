"""
automate_Christophorus-Anggit.py

Otomatisasi preprocessing dataset Telco Customer Churn.
Langkahnya sama persis dengan notebook Eksperimen_Christophorus-Anggit.ipynb,
hanya disusun ulang menjadi fungsi agar bisa dijalankan otomatis
(lokal maupun lewat GitHub Actions).

Cara pakai:
    python automate_Christophorus-Anggit.py \
        --input ../telco_churn_raw/Telco-Customer-Churn.csv \
        --output telco_churn_preprocessing
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Churn"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]


def load_data(path):
    """Membaca dataset mentah dari file CSV."""
    return pd.read_csv(path)


def clean_data(df):
    """Membersihkan data: buang ID, perbaiki tipe TotalCharges, hapus duplikat."""
    df = df.copy()

    # customerID hanya penanda unik, tidak membawa informasi untuk prediksi
    df = df.drop(columns=["customerID"])

    # TotalCharges tersimpan sebagai teks dan berisi 11 nilai kosong (" ").
    # Semua baris kosong itu punya tenure = 0 (pelanggan baru, belum ditagih),
    # jadi nilai yang masuk akal adalah 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def cap_outliers_iqr(df, cols):
    """Memotong (capping) nilai di luar batas IQR. Pada dataset ini tidak ada
    yang terpotong, tetapi langkah ini tetap dijalankan agar pipeline aman
    jika suatu saat data baru membawa outlier."""
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    return df


def encode_data(df):
    """Encoding: target dan kolom biner menjadi 0/1, sisanya one-hot."""
    df = df.copy()
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    for col in BINARY_COLS[1:]:
        df[col] = df[col].map({"No": 0, "Yes": 1})

    multi_cols = [
        c for c in df.columns
        if df[c].dtype == "object" or str(df[c].dtype) == "str"
    ]
    df = pd.get_dummies(df, columns=multi_cols, dtype=int)
    return df


def split_and_scale(df, test_size=0.2, random_state=42):
    """Split train/test (stratified), lalu standarisasi kolom numerik.
    Scaler di-fit HANYA pada data train agar tidak ada kebocoran data."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    return train, test


def preprocess(input_path, output_dir):
    """Pipeline lengkap: raw CSV -> train.csv dan test.csv siap latih."""
    df = load_data(input_path)
    df = clean_data(df)
    df = cap_outliers_iqr(df, NUMERIC_COLS)
    df = encode_data(df)
    train, test = split_and_scale(df)

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Train: {train.shape} | Test: {test.shape}")
    print(f"Disimpan ke: {output_dir}")
    return train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Telco Customer Churn")
    parser.add_argument(
        "--input", default="../telco_churn_raw/Telco-Customer-Churn.csv"
    )
    parser.add_argument("--output", default="telco_churn_preprocessing")
    args = parser.parse_args()
    preprocess(args.input, args.output)
