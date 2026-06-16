import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Konfigurasi
IMG_SIZE = (224, 224)
MODEL_PATH = 'bisindo_final.keras'

# Label kelas
CLASS_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', 'venv']

print("-" * 50)
print("DEMO PREDIKSI BISINDO")
print("-" * 50)

# Cek model
if not os.path.exists(MODEL_PATH):
    print(f"Model '{MODEL_PATH}' tidak ditemukan!")
    exit()

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model siap!\n")

# ============================================
# FUNGSI PREDIKSI
# ============================================

def predict_single(image_path, show_plot=True):
    """Prediksi satu gambar BISINDO"""
    
    # Baca gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"Gagal membaca: {image_path}")
        return None, None
    
    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_label = CLASS_LABELS[predicted_idx]
    
    # Top 5
    top5_idx = np.argsort(predictions[0])[-5:][::-1]
    top5 = [(CLASS_LABELS[i], predictions[0][i]) for i in top5_idx]
    
    # Tampilkan hasil
    if show_plot:
        print(f"\nFile: {os.path.basename(image_path)}")
        print(f"Hasil: {predicted_label} (confidence: {confidence:.2%})")
        print("\nTop 5:")
        for i, (label, prob) in enumerate(top5, 1):
            print(f"  {i}. {label}: {prob:.2%}")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(img_rgb)
        ax1.set_title('Gambar Input')
        ax1.axis('off')
        
        labels = [l for l, _ in top5]
        probs = [p for _, p in top5]
        colors = ['green' if i == 0 else 'skyblue' for i in range(len(top5))]
        ax2.barh(labels, probs, color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Confidence')
        ax2.set_title(f'Top 5: {predicted_label} ({confidence:.2%})')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return predicted_label, confidence

def predict_all():
    """Prediksi semua huruf A-Z dari dataset"""
    dataset_path = r'D:\SSD Sebelah\Perkuliahan\Kecerdasan Buatan\Dataset_Bisindo'
    
    if not os.path.exists(dataset_path):
        print(f"Dataset tidak ditemukan di: {dataset_path}")
        return
    
    print("\n" + "-" * 50)
    print("PREDIKSI SEMUA HURUF A-Z")
    print("-" * 50)
    
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']
    
    results = []
    
    for letter in letters:
        letter_path = os.path.join(dataset_path, letter)
        if not os.path.exists(letter_path):
            continue
        
        images = [f for f in os.listdir(letter_path) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not images:
            continue
        
        img_path = os.path.join(letter_path, images[0])
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_array = img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        idx = np.argmax(predictions[0])
        pred_label = CLASS_LABELS[idx]
        conf = np.max(predictions[0])
        
        is_correct = (letter == pred_label)
        results.append((letter, pred_label, conf, is_correct))
    
    # Tampilkan hasil
    print(f"\n{'Huruf':<8} {'Prediksi':<10} {'Confidence':<12} {'Status':<10}")
    print("-" * 50)
    
    correct = 0
    for letter, pred, conf, is_correct in results:
        status = "OK" if is_correct else "SALAH"
        print(f"{letter:<8} {pred:<10} {conf:.2%}      {status}")
        if is_correct:
            correct += 1
    
    print("-" * 50)
    print(f"\nAkurasi: {correct}/{len(results)} ({correct/len(results):.2%})")
    
    return results

def main():
    while True:
        print("\n" + "-" * 50)
        print("MENU DEMO")
        print("-" * 50)
        print("1. Prediksi satu gambar")
        print("2. Prediksi semua huruf A-Z")
        print("3. Keluar")
        print("-" * 50)
        
        choice = input("\nPilih (1-3): ").strip()
        
        if choice == "1":
            path = input("Masukkan path gambar: ").strip()
            if path:
                predict_single(path)
            
        elif choice == "2":
            predict_all()
            
        elif choice == "3":
            print("\nSelesai. Terima kasih!")
            break
            
        else:
            print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()