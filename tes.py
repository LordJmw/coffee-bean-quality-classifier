# test_dataset.py - LANJUTAN
import os

base_path = "data/training dataset"

print("=" * 50)
print("DEBUG DATASET LOADER")
print("=" * 50)

# 1. Cek base_path
print(f"\n1. Base path: {base_path}")
print(f"   Exists: {os.path.exists(base_path)}")

if os.path.exists(base_path):
    # 2. List folder
    all_items = os.listdir(base_path)
    folders = [f for f in all_items if os.path.isdir(os.path.join(base_path, f))]
    
    print(f"\n2. Semua folder ditemukan ({len(folders)}):")
    for f in sorted(folders):
        print(f"   - {f}")
    
    # 3. Cek DATASET_INFO vs folder asli
    print(f"\n3. Cek DATASET_INFO:")
    from utils.dataset_loader import DATASET_INFO
    
    for class_name, info in DATASET_INFO.items():
        folder_name = info['folder']
        folder_path = os.path.join(base_path, folder_name)
        exists = os.path.exists(folder_path)
        status = "✅" if exists else "❌"
        print(f"   {status} {class_name} -> '{folder_name}'")
    
    # 4. Test get_dataset_samples
    print(f"\n4. Hasil get_dataset_samples():")
    from utils.dataset_loader import get_dataset_samples
    samples = get_dataset_samples(base_path)
    print(f"   Total samples returned: {len(samples)}")
    
    if len(samples) < len(folders):
        print(f"\n   ❌ Folder yang TIDAK muncul:")
        for folder in sorted(folders):
            if folder not in samples:
                print(f"      - {folder}")
    else:
        print(f"   ✅ Semua folder muncul!")
        for name in sorted(samples.keys()):
            info = samples[name]
            print(f"      - {name} ({info['total_images']} gambar)")