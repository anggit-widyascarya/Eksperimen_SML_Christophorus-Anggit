# Eksperimen_SML_Christophorus-Anggit

Kriteria 1 submission Membangun Sistem Machine Learning (Dicoding).
Dataset: Telco Customer Churn (IBM), klasifikasi biner `Churn`.

| Path | Isi |
|---|---|
| `telco_churn_raw/` | Dataset mentah |
| `preprocessing/Eksperimen_Christophorus-Anggit.ipynb` | Eksperimen manual: loading, EDA, preprocessing |
| `preprocessing/automate_Christophorus-Anggit.py` | Versi otomatis dari preprocessing di notebook |
| `preprocessing/telco_churn_preprocessing/` | `train.csv` dan `test.csv` siap latih |
| `.github/workflows/preprocessing.yml` | GitHub Actions: menjalankan preprocessing otomatis |

Menjalankan lokal:

```bash
pip install -r requirements.txt
cd preprocessing
python automate_Christophorus-Anggit.py
```
