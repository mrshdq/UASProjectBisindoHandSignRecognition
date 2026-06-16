import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2

print("="*60)
print("PROYEK BISINDO")
print("="*60)
print(f"TensorFlow: {tf.__version__}")
print(f"OpenCV: {cv2.__version__}")

# Path dataset
data_dir = r'D:\SSD Sebelah\Perkuliahan\Kecerdasan Buatan\Dataset_Bisindo'
img_size = (224, 224)
batch = 16
seed = 42
epochs = 50

# Huruf yang susah
susah = ['C', 'D', 'F', 'L', 'O', 'P', 'Q', 'R', 'T', 'U', 'X', 'Y']
prioritas = ['D', 'P', 'U', 'X', 'Y']
bermasalah = ['H', 'I', 'M', 'B', 'E']  # Dari hasil training sebelumnya

if not os.path.exists(data_dir):
    print(f"Folder tidak ditemukan: {data_dir}")
    exit()

# Fungsi buat nambah data
def tambah_data(letter_path, letter, jumlah=8):
    files = [f for f in os.listdir(letter_path) 
             if f.endswith(('.jpg', '.png', '.jpeg')) 
             and not f.startswith('aug_')]
    
    if not files:
        return []
    
    print(f"  {letter}: {len(files)} gambar")
    hasil = []
    
    for f in files:
        path = os.path.join(letter_path, f)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        
        hasil.append((img, letter))
        
        for _ in range(jumlah):
            temp = img.copy()
            
            # Putar
            sudut = np.random.uniform(-35, 35)
            h, w = temp.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), sudut, 1)
            temp = cv2.warpAffine(temp, M, (w, h))
            
            # Terang/gelap
            terang = np.random.uniform(0.5, 1.5)
            temp = np.clip(temp * terang, 0, 1)
            
            # Flip
            if np.random.random() > 0.5:
                temp = cv2.flip(temp, 1)
            
            # Blur
            if np.random.random() > 0.6:
                temp = cv2.GaussianBlur(temp, (3, 3), 0)
            
            hasil.append((temp, letter))
    
    return hasil

# Load data
print("\nLoading dataset...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch,
    class_mode='categorical',
    subset='training',
    seed=seed,
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch,
    class_mode='categorical',
    subset='validation',
    seed=seed,
    shuffle=False
)

n_classes = len(train_gen.class_indices)
labels = list(train_gen.class_indices.keys())
indices = train_gen.class_indices

print(f"Jumlah kelas: {n_classes}")
print(f"Train: {train_gen.samples}")
print(f"Val: {val_gen.samples}")

# Bobot kelas
weights = {}
print("\nClass weights:")
for name, idx in indices.items():
    if name in prioritas:
        weights[idx] = 2.5
        print(f"  {name}: 2.5")
    elif name in bermasalah:
        weights[idx] = 2.0
        print(f"  {name}: 2.0")
    else:
        weights[idx] = 1.0
        if name != 'venv':
            print(f"  {name}: 1.0")

# Augmentasi
print("\n--- AUGMENTASI ---")
tambahan_data = []
tambahan_label = []

for letter in susah:
    path = os.path.join(data_dir, letter)
    if not os.path.exists(path):
        continue
    
    files = [f for f in os.listdir(path) 
             if f.endswith(('.jpg', '.png', '.jpeg')) 
             and not f.startswith('aug_')]
    
    jml = 5 if letter in prioritas else 4
    print(f"  {letter}: {len(files)} gambar")
    
    for f in files[:5]:
        p = os.path.join(path, f)
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        
        tambahan_data.append(img)
        tambahan_label.append(indices[letter])
        
        for _ in range(jml):
            temp = img.copy()
            sudut = np.random.uniform(-25, 25)
            h, w = temp.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), sudut, 1)
            temp = cv2.warpAffine(temp, M, (w, h))
            terang = np.random.uniform(0.6, 1.4)
            temp = np.clip(temp * terang, 0, 1)
            if np.random.random() > 0.5:
                temp = cv2.flip(temp, 1)
            tambahan_data.append(temp)
            tambahan_label.append(indices[letter])

# Khusus yang bermasalah
print("\n--- KHUSUS YANG BERMASALAH ---")
for letter in bermasalah:
    path = os.path.join(data_dir, letter)
    if not os.path.exists(path):
        continue
    hasil = tambah_data(path, letter, 8)
    for img, lbl in hasil:
        tambahan_data.append(img)
        tambahan_label.append(indices[lbl])

if len(tambahan_data) > 0:
    tambahan_data = np.array(tambahan_data)
    tambahan_label = to_categorical(tambahan_label, num_classes=n_classes)
    print(f"\nTotal data tambahan: {len(tambahan_data)}")

# Gabung data
print("\n--- GABUNG DATA ---")
x_train, y_train = [], []
for i in range(len(train_gen)):
    x_batch, y_batch = train_gen[i]
    x_train.extend(x_batch)
    y_train.extend(y_batch)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.concatenate([x_train, tambahan_data])
y_train = np.concatenate([y_train, tambahan_label])

print(f"Total train: {len(x_train)} gambar")

# Model
print("\n--- BANGUN MODEL ---")
base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Training
print("\n--- TRAINING ---")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(
    x_train, y_train,
    validation_data=val_gen,
    epochs=epochs,
    batch_size=batch,
    class_weight=weights,
    callbacks=callbacks,
    verbose=1
)

model.save('bisindo_final.keras')

# Plot
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot.png', dpi=150)
plt.show()

# Evaluasi
print("\n--- EVALUASI ---")
Y_pred = model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(val_gen.classes, y_pred)
acc_per_class = cm.diagonal() / cm.sum(axis=1)

print("\nClassification Report:")
print(classification_report(val_gen.classes, y_pred,
                            target_names=labels, zero_division=0))

# Confusion matrix
plt.figure(figsize=(18, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={'size': 8})
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('confusion.png', dpi=150)
plt.show()

# Akurasi per huruf
print("\n--- AKURASI PER HURUF ---")
for i, label in enumerate(labels):
    if label != 'venv':
        print(f"  {label}: {acc_per_class[i]:.2%}")

# Kesimpulan
rata = np.mean(acc_per_class)
print(f"\nRata-rata akurasi: {rata:.2%}")
print(f"Best val: {max(history.history['val_accuracy']):.2%}")

if rata >= 0.95:
    print("\nModel siap pakai!")
elif rata >= 0.85:
    print("\nModel cukup baik.")
else:
    print("\nModel perlu perbaikan.")

print("\nSelesai!")