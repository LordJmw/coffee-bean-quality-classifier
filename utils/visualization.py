import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Tambahan untuk handle legend yang aman
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import streamlit as st

CLASS_NAMES = [
    'Normal',
    'Withered',
    'Partial Sour',
    'Broken',
    'Dry Cherry',
    'Severe Insect Damage'
]

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

@st.cache_data
def load_visualization_data():
    df = pd.read_csv("data/features_dataset.csv")
    return df

def create_decision_boundary_plot(model, scaler, user_features=None):
    # Load dataset
    df = load_visualization_data()

    X = df[FEATURES].values
    y = df['class_num'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Fitur yang divisualisasikan
    idx_x = FEATURES.index('green_ratio')
    idx_y = FEATURES.index('area')

    # Feature importance
    importances = model.feature_importances_
    imp_x = importances[idx_x]
    imp_y = importances[idx_y]

    # Median fitur lain
    medians = np.median(X_train, axis=0)

    # Range plot
    x_min, x_max = X[:, idx_x].min(), X[:, idx_x].max()
    y_min, y_max = X[:, idx_y].min(), X[:, idx_y].max()

    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    # Resolusi grid diubah ke 200 agar render di Streamlit lebih cepat tanpa mengurangi visual
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_margin, x_max + x_margin, 200),
        np.linspace(y_min - y_margin, y_max + y_margin, 200)
    )

    # Grid 10D
    grid_x_flat = xx.ravel()
    grid_y_flat = yy.ravel()

    grid_10d = np.zeros((len(grid_x_flat), 10))

    for i in range(10):
        grid_10d[:, i] = medians[i]

    grid_10d[:, idx_x] = grid_x_flat
    grid_10d[:, idx_y] = grid_y_flat

    # Prediksi
    grid_scaled = scaler.transform(grid_10d)
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))

    colors = [
        '#FF9999', # Normal
        '#66B3FF', # Withered
        '#99FF99', # Partial Sour
        '#FFCC99', # Broken
        '#C2C2F0', # Dry Cherry
        '#FFD700'  # Severe Insect Damage
    ]
    cmap = ListedColormap(colors)

    # 1. Gambar Background (Prediksi Model)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    # 2. Gambar Titik (Data Aktual)
    # Menggunakan c=y_test dengan vmin & vmax agar mapping warnanya mengunci ke urutan 0-5
    ax.scatter(
        X_test[:, idx_x],
        X_test[:, idx_y],
        c=y_test,
        cmap=cmap,
        vmin=0,
        vmax=len(CLASS_NAMES)-1,
        edgecolors='k',
        s=40
    )

    # --- PERBAIKAN LEGEND AMAN ---
    # Membuat handles legend secara konsisten tanpa peduli apakah kelasnya ada di test set atau tidak
    handles = [mpatches.Patch(color=colors[i], label=CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))]
    labels = list(CLASS_NAMES)

    # 3. TITIK INPUT USER (Jika ada)
    if user_features is not None:
        user_green_ratio = user_features.get('green_ratio', 0)
        user_area = user_features.get('area', 0)

        user_marker = ax.scatter(
            user_green_ratio,
            user_area,
            color='black',
            s=250,
            marker='*',
            edgecolors='white',
            linewidths=1.5
        )
        # Masukkan plot bintang ke list legend
        handles.append(user_marker)
        labels.append('Input User saat ini (*)')

    # 4. SATUKAN SEMUA DI LEGEND UTAMA
    ax.legend(
        handles=handles,
        labels=labels,
        title="Keterangan Kelas:\n(Bg = Prediksi Model | Titik = Data Aktual)",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=True
    )

    ax.set_title(
        f"Decision Boundary Random Forest\n"
        f"Green Ratio (Imp: {imp_x:.3f}) vs "
        f"Area (Imp: {imp_y:.3f})"
    )
    ax.set_xlabel("Green Ratio")
    ax.set_ylabel("Area")
    
    plt.tight_layout()
    return fig