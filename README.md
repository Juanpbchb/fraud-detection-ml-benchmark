# 🕵️ Online Payment Fraud Detection with Machine Learning

An experimental study, using both **supervised and unsupervised learning**, of detecting fraud in mobile payment transactions and classifying it by type. The goal is not just to build a good classifier but to measure **how each preprocessing decision** (scaling, outlier handling and class balancing) affects different families of models when the classes are **extremely imbalanced** (~0.13% fraud).

> Universidad Nacional de Colombia – Medellín · Facultad de Minas · Introduction to Artificial Intelligence (2025)

---

## 📌 Overview

| Aspect | Details |
|---|---|
| **Task** | Multiclass classification (5 classes): legitimate vs. 4 fraud types |
| **Data** | [Online Payments Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset): simulated, ~6.3M rows |
| **Sample used** | 8,000 records (stratified sampling on `isFraud`) |
| **Experimental design** | 8 dataset variants = 2 (scaling) × 2 (outliers) × 2 (balancing) |
| **Supervised models** | Decision Tree, KNN, SVM (RBF, OvR/OvO), Neural Network (Keras) |
| **Unsupervised models** | K-Means, DBSCAN |
| **Main metric** | F1-score (macro and weighted), plus ROC-AUC and PR-AUC (OvR) |
| **Stack** | Python, pandas, NumPy, scikit-learn, imbalanced-learn, TensorFlow/Keras, matplotlib, seaborn |

---

## 🗂️ Repository Structure

```text
.
├── Preprocesamiento_del_Dataset.ipynb   # Sampling, cleaning, label engineering and generation of the 8 variants
├── Arboles_de_Decision.ipynb            # DecisionTreeClassifier + feature importance + ROC/PR
├── supervisado_knn.ipynb                # KNN with hyperparameter search (k, weights, p)
├── SVM.ipynb                            # RBF-kernel SVC, OvR vs OvO, 2D decision boundaries
├── RedNeuronal.ipynb                    # Keras MLP (L2 + Dropout + EarlyStopping), 2D PCA
├── no_supervisado.ipynb                 # K-Means (elbow + silhouette) and DBSCAN (k-distance)
├── Tablas de resultados_.xlsx           # Consolidated metrics table per model and variant
└── Datasets/
    ├── df_CC_Desbalanceado.csv          # Clean base dataset (one-hot, unscaled, imbalanced)
    ├── Datos_Supervisado/               # 8 variants WITH the `fraud_category_5_classes` label
    │   ├── df_CC_{NO|SI}_ED_Desbalanceado_{Con|Sin}_Outlier(s).csv
    │   └── df_CC_{NO|SI}_ED_Balanceado_{Con|Sin}_Outlier(s).csv
    └── Datos__No_Supervisado/           # The same 8 variants WITHOUT the target variable
        └── ..._Sin_Variable_Objetivo.csv
```

### Dataset Naming Convention

File names are in Spanish:

| Token | Meaning |
|---|---|
| `CC` | **Categorical encoding** applied (one-hot of `type`); present in every variant |
| `SI_ED` / `NO_ED` | With / without **data scaling** of the continuous features |
| `Balanceado` / `Desbalanceado` | Balanced (SMOTE + undersampling) / imbalanced |
| `Con_Outliers` / `Sin_Outliers` | Outliers kept / removed |
| `Sin_Variable_Objetivo` | Target variable removed (for unsupervised learning) |

---

## 🔧 Preprocessing Pipeline

```mermaid
flowchart LR
    A["Kaggle dataset ~6.3M rows"] --> B["Stratified sample 8k"]
    B --> C["Drop: step, isFlaggedFraud, nameOrig, nameDest"]
    C --> D["Nulls and duplicates (36 removed)"]
    D --> E["5-class target"]
    E --> F["One-hot encode type"]
    F --> G{"Outliers"}
    G --> H{"Scaling"}
    H --> I{"Balancing"}
    I --> J["8 supervised datasets"]
    J --> K["8 unsupervised datasets (no y)"]
```

**Key decisions:**

- **Stratified subsampling** from 6.3M to 8k rows to keep computation manageable. The fraud rate stays almost the same (0.129% → 0.125%).
- **Columns dropped because of bias or leakage risk:**
  - `isFlaggedFraud`: output of the simulator's existing rule-based system. Keeping it would teach the model to copy those rules.
  - `nameOrig` / `nameDest`: high-cardinality IDs, which invite memorization.
  - `step`: synthetic time index, not useful since this is not time-series modeling.
- **Target engineering** (`fraud_category_5_classes`): fraud only occurs in `TRANSFER` and `CASH_OUT`, so fraud was split by **type** and **amount**, using the median of `amount` as the threshold:
  1. Legitimate
  2. Fraud – Transfer – Low amount
  3. Fraud – Transfer – High amount
  4. Fraud – Cash Out – Low amount
  5. Fraud – Cash Out – High amount
- **Outliers**: detected on `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest` and `newbalanceDest` (boxplots).
- **Balancing**: creating thousands of synthetic samples per class from ~10 real frauds would lead to memorization. As a compromise, **SMOTE** oversamples each minority class to 1,200 samples and **undersampling** reduces the majority class to 1,200.

---

## 🤖 Models and Configuration

All supervised models use a **stratified** `train_test_split` (70/30, `random_state=42`).

| Model | Relevant configuration |
|---|---|
| **Decision Tree** | `DecisionTreeClassifier` with `class_weight='balanced'` on imbalanced variants; feature importance and root-node analysis |
| **KNN** | Manual grid: `n_neighbors ∈ {3,5,7}`, `weights ∈ {uniform, distance}`, `p ∈ {1,2}` |
| **SVM** | `SVC(kernel='rbf', C=1)`, compared **One-vs-Rest and One-vs-One**; OvR was kept because it was more stable and scales better. 2D decision boundaries on `amount` × `oldbalanceOrg` |
| **Neural Network** | MLP `Dense(64, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(5, softmax)`, L2 = 0.01, Adam, `categorical_crossentropy`, `EarlyStopping(patience=10, restore_best_weights=True)`, 100 epochs, batch size 32, `validation_split=0.2`. Inputs: `amount`, `oldbalanceOrg` |
| **K-Means** | k chosen with the elbow method, evaluated with the Silhouette Score |
| **DBSCAN** | `eps` chosen from the k-distance plot (≈10⁵–10⁶ on unscaled data, 0.1–1 on scaled data) |

Visualizations: confusion matrices, multiclass ROC and Precision-Recall curves (OvR), tree plots, feature importance, decision boundaries and 2D PCA projections.

---

## 📊 Results

### F1-score (weighted) per model and variant

| # | Scaling | Outliers | Balancing | Tree | KNN | SVM | Neural Net |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | No | No | No | 0.22* | **0.999** | 0.89 | 0.99† |
| 2 | No | No | Yes | 0.98 | **0.996** | 0.92 | 0.75 |
| 3 | No | Yes | No | 0.42* | **0.968** | 0.77 | 0.99† |
| 4 | No | Yes | Yes | **0.98** | 0.973 | 0.69 | 0.57 |
| 5 | Yes | No | No | 0.22* | **0.997** | 0.89 | 0.99† |
| 6 | Yes | No | Yes | 0.97 | **0.995** | 0.92 | 0.75 |
| 7 | Yes | Yes | No | 0.41* | **0.972** | 0.77 | 0.99† |
| 8 | Yes | Yes | Yes | **0.98** | 0.969 | 0.74 | 0.57 |

\* For the Decision Tree on imbalanced variants, the **macro-F1** is reported because it shows how poorly the model handles the minority classes.
† Misleading: the network predicts **only the "Legitimate" class**, so the weighted F1 is driven entirely by the majority class.

### Key Findings

- **Accuracy is not useful here.** With about one fraud case per class in the test set, a model that always predicts "Legitimate" scores above 97%. The analysis therefore relies on macro-F1, per-class recall and PR curves.
- **Balancing is the most important factor** for the Decision Tree (macro-F1 from 0.22 to 0.98). It also stops the neural network from predicting only the majority class, although it lowers the network's weighted F1. Likely reasons are noise added by SMOTE and a model that uses only 2 input features.
- **KNN was the most robust model** across all 8 variants (F1 ≈ 0.97–1.00). Its best configuration on imbalanced data was `k=5, weights='distance', p=1`.
- **Scaling made little difference** for any model, even the distance-based KNN and SVM. This suggests the relevant features already separate the classes well.
- **Outliers help tree-based models** (their F1 goes up), but they hurt SVM and the neural network, which are more sensitive to extreme values.
- **The most common confusion** is between frauds of the same type with different amounts (high vs. low). This is expected, since the median threshold is an artificial boundary.
- **Most important features** (Decision Tree): `oldbalanceOrg`, `oldbalanceDest`/`newbalanceDest` and `type`, all related to how the origin and destination balances change.
- **Clustering**: K-Means kept a Silhouette score above 0.7 in most variants and did best when scaling and balancing were combined. DBSCAN varied much more and was sensitive to outliers, but it was useful for separating them out as noise.

---

## 🚀 Getting Started

The notebooks were developed in **Google Colab** and load the CSV files from Google Drive.

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow matplotlib seaborn openpyxl
```

1. Clone the repository and open the notebooks in Jupyter or Colab.
2. Replace the `'/content/drive/Shareddrives/...'` paths and the `drive.mount(...)` calls with local paths to `Datasets/`.
3. Suggested order: `Preprocesamiento_del_Dataset` → supervised models → `no_supervisado`.

> [!NOTE]
> The processed CSV files are included, so you don't need the original dataset to reproduce the models. Re-running the preprocessing requires the full Kaggle CSV.

> [!NOTE]
> Notebook comments and outputs are written in Spanish.

---

## ⚠️ Limitations and Future Work

- **Very few real fraud cases** (~10 in 8k rows). Metrics on the imbalanced test sets have high variance, and in the balanced variants most fraud samples are synthetic (SMOTE), which may inflate results.
- **Possible data leakage from balancing**: SMOTE was applied before the train/test split. It should be applied **only to the training set**, for example with an `imblearn.Pipeline` inside cross-validation.
- The data is **simulated** (PaySim-like), so the patterns may not carry over to real-world transactions.
- Next steps: use more data or the full dataset, stratified cross-validation, boosting models (XGBoost/LightGBM), per-class threshold tuning, cost-sensitive learning and anomaly detection (Isolation Forest, autoencoders).

---

## 👥 Authors

- Juan Pablo Chamorro Bolaños
- Juan José Sánchez Mejía
- Gilary Carolina Sierra Céspedes

Advisor: Prof. Jaime Alberto Guzmán Luna
