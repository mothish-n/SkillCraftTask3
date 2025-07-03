import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# === Configuration ===
DATADIR = r"C:\Users\mothi\Downloads\archive (4)\dataset\train"
CATEGORIES = ['cat', 'dog']
IMG_SIZE = 64
LIMIT = 1000  # limit per class for training

def create_data(limit_per_class=LIMIT):
    data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        count = 0
        for img in tqdm(os.listdir(path), desc=f"Loading {category}"):
            if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    continue
                resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_img.flatten(), class_num])
                count += 1
                if count >= limit_per_class:
                    break
            except Exception as e:
                print(f"Error loading {img}: {e}")
    return data

def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear')
    print("Training SVM model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ SVM Accuracy: {acc * 100:.2f}%")
    return model

def predict_image(model, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Image not found or unreadable.")
        return
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
    prediction = model.predict(img_resized)[0]

    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Prediction: {'Dog 🐶' if prediction == 1 else 'Cat 🐱'}")
    plt.show()

if __name__ == "__main__":
    print("✅ Starting Cats vs Dogs SVM classification...\n")
    
    data = create_data()
    print(f"✅ Total samples loaded: {len(data)}")
    
    if len(data) < 2:
        print("❌ Not enough data to train. Please check your dataset.")
        exit()
    
    X = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])
    
    model = train_svm(X, y)
    
    # 🔁 Example test image (change the path if needed)
    sample_img = os.path.join(DATADIR, "cat", "cat.123.jpg")
    predict_image(model, sample_img)
