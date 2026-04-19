# utils/dataset_loader.py
import os

# utils/dataset_loader.py - DATASET_INFO yang SUDAH DISESUAIKAN

DATASET_INFO = {
    # Grade 1
    'Normal': {
        'label': 'Normal',
        'grade': 1,
        'description': 'Biji normal, tidak cacat',
        'folder': 'Normal'  # Persis sama
    },
    
    # Grade 2
    'Fade': {
        'label': 'Fade',
        'grade': 2,
        'description': 'Biji pudar/kusam',
        'folder': 'Fade'
    },
    'Withered': {
        'label': 'Withered',
        'grade': 2,
        'description': 'Biji layu/keriput',
        'folder': 'Withered'
    },
    
    # Grade 3
    'Partial Black': {
        'label': 'Partial Black',
        'grade': 3,
        'description': 'Hitam sebagian',
        'folder': 'Partial Black'
    },
    'Partial Sour': {
        'label': 'Partial Sour',
        'grade': 3,
        'description': 'Asam sebagian',
        'folder': 'Partial Sour'
    },
    'Immature': {
        'label': 'Immature',
        'grade': 3,
        'description': 'Biji muda',
        'folder': 'Immature'
    },
    'Shell': {
        'label': 'Shell',
        'grade': 3,
        'description': 'Cangkang kosong',
        'folder': 'Shell'
    },
    'Husk': {
        'label': 'Husk',
        'grade': 3,
        'description': 'Kulit tanduk',
        'folder': 'Husk'
    },
    
    # Grade 4
    'Broken': {
        'label': 'Broken',
        'grade': 4,
        'description': 'Pecah',
        'folder': 'Broken'
    },
    'Cut': {
        'label': 'Cut',
        'grade': 4,
        'description': 'Terpotong',
        'folder': 'Cut'
    },
    'Slight Insect Damage': {
        'label': 'Slight Insect',
        'grade': 4,
        'description': 'Lubang hama ringan',
        'folder': 'Slight Insect Damage'
    },
    'Dry Cherry': {
        'label': 'Dry Cherry',
        'grade': 4,
        'description': 'Kulit kering',
        'folder': 'Dry Cherry'
    },
    'Floater': {
        'label': 'Floater',
        'grade': 4,
        'description': 'Biji mengapung',
        'folder': 'Floater'
    },
    'Parchment': {
        'label': 'Parchment',
        'grade': 4,
        'description': 'Kulit ari',
        'folder': 'Parchment'
    },
    'Full Sour': {
        'label': 'Full Sour',
        'grade': 4,
        'description': 'Asam total',
        'folder': 'Full Sour'
    },
    
    # Grade 5
    'Full Black': {
        'label': 'Full Black',
        'grade': 5,
        'description': 'Hitam total',
        'folder': 'Full Black'
    },
    'Fungus Damage': {
        'label': 'Fungus',
        'grade': 5,
        'description': 'Jamur',
        'folder': 'Fungus Damage'
    },
    'Severe Insect Damage': {
        'label': 'Severe Insect',
        'grade': 5,
        'description': 'Lubang hama parah',
        'folder': 'Severe Insect Damage'
    }
}

def get_dataset_samples(base_path="data/training_dataset"):
    """
    Mendapatkan daftar sampel dataset yang tersedia
    Dataset: 18 kelas, single bean per gambar
    """
    samples = {}
    
    if os.path.exists(base_path):
        for class_name, info in DATASET_INFO.items():
            folder_path = os.path.join(base_path, info['folder'])
            
            if os.path.exists(folder_path):
                # Ambil semua gambar dalam folder
                images = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if images:
                    # Sort gambar agar urut
                    images.sort()
                    
                    samples[class_name] = {
                        'folder': folder_path,
                        'images': images[:30],  # Batasi 30 gambar pertama untuk dropdown
                        'total_images': len(images),
                        'grade': info['grade'],
                        'label': info['label'],
                        'description': info['description']
                    }
    
    return samples

def get_dataset_statistics(base_path="data/training_dataset"):
    """
    Mendapatkan statistik lengkap dataset
    """
    stats = {
        'total_images': 0,
        'per_grade': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        'per_class': {}
    }
    
    if os.path.exists(base_path):
        for class_name, info in DATASET_INFO.items():
            folder_path = os.path.join(base_path, info['folder'])
            
            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                count = len(images)
                grade = info['grade']
                
                stats['total_images'] += count
                stats['per_grade'][grade] += count
                stats['per_class'][class_name] = {
                    'count': count,
                    'grade': grade,
                    'label': info['label']
                }
    
    return stats

def load_image_from_dataset(image_path):
    """
    Load gambar dari path dataset
    """
    import cv2
    
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
    return None