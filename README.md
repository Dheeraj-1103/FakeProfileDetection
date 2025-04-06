---

## 🔍 Fake Profile Detection using ML Ensemble

This project is a **machine learning pipeline** for detecting fake Twitter profiles using metadata. It includes preprocessing, exploratory data analysis, training multiple classifiers, and evaluating an ensemble model for optimal performance.

---

### 📁 Files in This Repository

```
.
├── users.csv                # Twitter user metadata (original profiles)
├── fusers.csv               # Twitter user metadata (fake profiles)
├── FakeProfileDetection.py # Main machine learning script
└── README.md                # Project overview and instructions
```

---

### 🧠 Objective

The script uses labeled Twitter metadata to classify profiles as either **real (`INT`)** or **fake (`E13`)** using various models, then combines the best ones using an ensemble strategy.

---

### ⚙️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Dheeraj-1103/FakeProfileDetection.git
   cd FakeProfileDetection
   ```

2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

3. Make sure the following files are present in the directory:
   - `users.csv`
   - `fusers.csv`
   - `FakeProfileDetection.py`

4. Run the pipeline:
   ```bash
   python FakeProfileDetection.py
   ```

---

### 📊 Pipeline Overview

#### 1. **Data Loading & Preprocessing**
   - Combines `users.csv` and `fusers.csv`
   - Drops irrelevant metadata
   - Encodes categorical features
   - Converts missing descriptions to binary

#### 2. **Visualization**
   - Class distribution plot
   - Correlation heatmap
   - Pairplots and boxplots of selected features

#### 3. **Model Training**
   - Algorithms used:
     - Random Forest
     - SVM
     - Multinomial Naive Bayes
     - K-Nearest Neighbors
   - Special scaling for Naive Bayes

#### 4. **Ensemble Voting**
   - Hard voting between Random Forest, SVM, and KNN

#### 5. **Evaluation**
   - Accuracy, classification reports, and confusion matrices

---

### 💾 Saving the Ensemble Model

To save the trained ensemble model, add the following at the end of the script:

```python
import joblib
joblib.dump(voting_clf, "fake_profile_model.pkl")
```

To load it later:

```python
voting_clf = joblib.load("fake_profile_model.pkl")
```

---

### 🛠 Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

---
