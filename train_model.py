"""
STEP 2: Training Model Random Forest
Menggunakan fitur dari CSV hasil ekstraksi
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi
CSV_PATH = "data/features_dataset.csv"
MODEL_PATH = "models/coffee_bean_rf.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
SCALER_PATH = "models/scaler.pkl"

# Nama kelas (urutan sesuai label_num di CSV)
CLASS_NAMES = ['Normal', 'Withered', 'Partial Sour', 'Broken', 'Dry Cherry', 'Severe Insect Damage']

# Fitur yang akan digunakan untuk training
FEATURES = [
    'area',
    'circularity',
    'solidity',
    'extent',
    'aspect_ratio',
    'holes_count',
    'center_cut_lines',
    'mean_intensity',
    'red_ratio',
    'green_ratio'
]


def load_and_prepare_data():
    """Load CSV dan siapkan X, y"""
    print("=" * 60)
    print("STEP 2: Training Random Forest")
    print("=" * 60)
    
    df = pd.read_csv(CSV_PATH)
    print(f"📊 Total data: {len(df)} sampel")
    print(f"\nDistribusi kelas:")
    for class_name in CLASS_NAMES:
        count = len(df[df['class_name'] == class_name])
        print(f"   {class_name}: {count} sampel")
    
    # Ambil fitur X dan target y
    X = df[FEATURES].values
    y = df['class_num'].values
    
    # Cek missing values
    if pd.isnull(X).any():
        print("\n⚠️ Ada missing values, akan diisi dengan median")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    print(f"\n✅ Fitur yang digunakan: {FEATURES}")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    return X, y, df


def train_model(X, y):
    """Latih Random Forest dengan validasi"""
    # Split data (80% train, 20% test, stratify agar distribusi kelas seimbang)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Split data:")
    print(f"   Training: {len(X_train)} sampel")
    print(f"   Testing: {len(X_test)} sampel")
    
    # Standardisasi fitur (penting untuk beberapa algoritma, untuk Random Forest tidak wajib tapi membantu)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest dengan parameter yang sudah disesuaikan untuk data kecil
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation (5 fold)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\n📈 Cross-validation (5-fold):")
    print(f"   Mean accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"   Per fold: {cv_scores}")
    
    # Training
    print("\n🔄 Training model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test set accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print("     " + " ".join([f"{c[:8]:<8}" for c in CLASS_NAMES]))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i][:8]:<8} " + " ".join([f"{val:<8}" for val in row]))
    
    return model, scaler, X_test_scaled, y_test, y_pred, cm


def plot_confusion_matrix(cm, save_path="models/confusion_matrix.png"):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n📁 Confusion matrix disimpan di: {save_path}")
    plt.close()


def show_feature_importance(model, feature_names):
    """Tampilkan feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n⭐ Feature Importance (pengaruh fitur terhadap prediksi):")
    print("-" * 40)
    for i in range(len(feature_names)):
        print(f"   {i+1}. {feature_names[indices[i]]:18s}: {importances[indices[i]]:.4f} ({importances[indices[i]]*100:.1f}%)")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importances[indices], align='center')
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest - Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    print(f"\n📁 Feature importance plot disimpan di: models/feature_importance.png")
    plt.close()


def save_model(model, scaler):
    """Simpan model dan scaler"""
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"\n💾 Model disimpan di: {MODEL_PATH}")
    print(f"💾 Scaler disimpan di: {SCALER_PATH}")


def main():
    import os
    
    # Load data
    X, y, df = load_and_prepare_data()
    
    # Train model
    model, scaler, X_test, y_test, y_pred, cm = train_model(X, y)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    # Show feature importance
    show_feature_importance(model, FEATURES)
    
    # Save model
    save_model(model, scaler)
    
    print("\n" + "=" * 60)
    print("✅ STEP 2 SELESAI! Model siap digunakan.")
    print("=" * 60)
    
    return model, scaler


if __name__ == "__main__":
    main()