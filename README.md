# 🖐️ BISINDO Hand Sign Recognition

## Pengenalan Huruf Bahasa Isyarat Indonesia menggunakan Convolutional Neural Network

---

### 📌 Deskripsi Proyek
Proyek ini bertujuan untuk mengenali dan mengklasifikasikan **27 huruf** dalam Bahasa Isyarat Indonesia (BISINDO) menggunakan **Convolutional Neural Network (CNN)** dengan arsitektur **MobileNetV2**.

**Akurasi yang dicapai: 100%** pada data validasi! 🏆

---

### 🎯 Tujuan
- Membantu penyandang tunarungu dalam berkomunikasi
- Mengembangkan sistem pengenalan isyarat tangan berbasis AI
- Menerapkan transfer learning dengan MobileNetV2

---

### 📊 Dataset
| Properti | Keterangan |
|----------|------------|
| **Jumlah data** | 312 gambar |
| **Jumlah kelas** | 27 (A-Z + venv) |
| **Split data** | 80% training, 20% validation |
| **Augmentasi** | Rotasi, brightness, flip, blur |

**Struktur Folder:**
Dataset_Bisindo/
├── A/
│ ├── body dot (1).jpg
│ ├── body dot (2).jpg
│ ├── white wall (1).jpg
│ └── ...
├── B/
│ └── ...
└── Z/
└── ...

---

### 🏗️ Arsitektur Model
MobileNetV2 (pretrained - frozen)
↓
GlobalAveragePooling2D
↓
BatchNormalization
↓
Dense 512 (ReLU) + L2 Regularization
↓
BatchNormalization + Dropout 0.5
↓
Dense 256 (ReLU) + L2 Regularization
↓
BatchNormalization + Dropout 0.4
↓
Dense 128 (ReLU) + L2 Regularization
↓
BatchNormalization + Dropout 0.3
↓
Dense 27 (Softmax)


| Parameter | Nilai |
|-----------|-------|
| **Total Parameter** | 3,090,267 |
| **Trainable** | 827,931 |
| **Learning Rate** | 0.0003 |
| **Epochs** | 50 |
| **Batch Size** | 16 |

---

### 📈 Hasil Evaluasi

| Metrik | Hasil |
|--------|-------|
| **Validation Accuracy** | **100%** |
| **Average Accuracy** | **100%** |
| **Precision** | 1.00 |
| **Recall** | 1.00 |
| **F1-Score** | 1.00 |

#### Confusion Matrix
![Confusion Matrix](confusion.png)

#### Accuracy & Loss Plot
![Plot](plot.png)

#### Akurasi per Kelas
| Huruf | Akurasi | Status |
|-------|---------|--------|
| A-Z | 100% | ✅ |
| H (sebelumnya bermasalah) | 100% | ✅ FIXED! |
| I, M, B, E | 100% | ✅ FIXED! |

---

### 🚀 Cara Menjalankan

#### 1. Clone Repository
```bash
git clone https://github.com/username/bisindo-hand-sign-recognition.git
cd bisindo-hand-sign-recognition
