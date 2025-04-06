# ML Classification Script with Preprocessing, Visualization, and Ensemble Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------- Data Loading & Cleaning --------------------
def load_and_clean_data(fusers_path: str, users_path: str) -> pd.DataFrame:
    df1 = pd.read_csv(users_path)
    df2 = pd.read_csv(fusers_path)
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

    # Drop irrelevant or redundant columns
    cols_to_drop = [
        "created_at", "profile_background_image_url", "lang", "url",
        "profile_text_color", "profile_background_image_url_https", "profile_banner_url",
        "id", "protected", "verified", "profile_image_url_https", "time_zone",
        "location", "profile_use_background_image", "default_profile_image",
        "profile_image_url", "geo_enabled", "fav_number", "updated",
        "profile_link_color", "profile_background_color", "utc_offset",
        "name", "screen_name"
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Binary transformation
    df["description"] = df["description"].notnull().astype(int)
    df["default_profile"] = df["default_profile"] == 1
    df["profile_background_tile"] = df["profile_background_tile"] == 1
    df["profile_sidebar_border_color"] = (df["profile_sidebar_border_color"] == "C0DEED").astype(int)
    df["profile_sidebar_fill_color"] = (df["profile_sidebar_fill_color"] == "DDEEF6").astype(int)

    # Encode target variable
    df["dataset"] = df["dataset"].map({"E13": 1, "INT": 0}).astype(int)

    return df

# -------------------- Data Visualization --------------------
def visualize_data(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="dataset")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    corr = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Pairplot for selected features
    selected_features = df.drop(columns="dataset").columns[:5].tolist()  # first 5 features for clarity
    sns.pairplot(df[selected_features + ['dataset']], hue='dataset', diag_kind='kde')
    plt.suptitle("Pairplot of Selected Features", y=1.02)
    plt.show()

    # Boxplot of each feature by class
    for col in selected_features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='dataset', y=col, data=df)
        plt.title(f"Boxplot of {col} by Class")
        plt.show()

# -------------------- Model Training & Evaluation --------------------
def train_models(X_train, X_test, y_train, y_test):
    # Separate scaler for Naive Bayes
    nb_scaler = MinMaxScaler()
    X_train_nb = nb_scaler.fit_transform(X_train)
    X_test_nb = nb_scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    for name, model in models.items():
        if name == "Naive Bayes":
            model.fit(X_train_nb, y_train)
            predictions = model.predict(X_test_nb)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

    return models

# -------------------- Ensemble Learning --------------------
def ensemble_learning(models, X_train, X_test, y_train, y_test):
    # Naive Bayes uses different scaling, so we exclude it
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', models["SVM"]),
            ('random_forest', models["Random Forest"]),
            ('knn', models["KNN"])
        ],
        voting='hard'
    )
    voting_clf.fit(X_train, y_train)
    ensemble_predictions = voting_clf.predict(X_test)

    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, ensemble_predictions))

# -------------------- Main Pipeline --------------------
def main():
    df = load_and_clean_data("fusers.csv", "users.csv")
    visualize_data(df)

    X = df.drop(columns=["dataset"])
    y = df["dataset"]

    # Scale features for most models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    models = train_models(X_train, X_test, y_train, y_test)
    ensemble_learning(models, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
