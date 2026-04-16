# utils/dataset_loader.py

DATASET_INFO = {
    'Normales.jpg': {'label': 'Good', 'grade': 1, 'description': 'Biji normal'},
    'Concha.jpg': {'label': 'Shell', 'grade': 3, 'description': 'Cangkang kosong'},
    'Inmaduro.jpg': {'label': 'Immature', 'grade': 3, 'description': 'Biji muda'},
    'PMordidoCortado.jpg': {'label': 'Broken', 'grade': 4, 'description': 'Pecah/terpotong'},
    'BrocadoLeve.jpg': {'label': 'Slight Insect', 'grade': 4, 'description': 'Lubang ringan'},
    'BrocadoSevero.jpg': {'label': 'Severe Insect', 'grade': 5, 'description': 'Lubang parah'},
    'DXHongo.jpg': {'label': 'Fungus', 'grade': 5, 'description': 'Jamur'},
    'Negros.jpg': {'label': 'Full Black', 'grade': 5, 'description': 'Hitam total'},
    'MarronAVinagre.jpg': {'label': 'Sour', 'grade': 5, 'description': 'Asam/fermentasi'},
    'CerezaSeca.jpg': {'label': 'Dried Cherry', 'grade': 5, 'description': 'Kulit kering'},
    'Pergamino.jpg': {'label': 'Parchment', 'grade': 5, 'description': 'Kulit tanduk'}
}

def get_dataset_samples(base_path="data/raw"):
    """Load informasi dataset"""
    samples = {}
    for filename, info in DATASET_INFO.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            samples[info['description']] = {
                'path': filepath,
                'grade': info['grade'],
                'label': info['label']
            }
    return samples