import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Hàm trích xuất đặc trưng và lưu tên ảnh
def extract_features_and_filenames(directory, model):
    features = []
    filenames = []
    
    # Lấy danh sách các ảnh trong thư mục
    img_filenames = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Lặp qua tất cả các ảnh trong thư mục với tqdm
    for img_filename in tqdm(img_filenames, desc="Processing images"):
        image_path = os.path.join(directory, img_filename)

        # Load ảnh và thay đổi kích thước về 224x224
        img = image.load_img(image_path, target_size=(224, 224))

        # Chuyển ảnh thành mảng numpy
        img_array = image.img_to_array(img)

        # Thêm chiều batch cho ảnh
        img_array = np.expand_dims(img_array, axis=0)

        # Tiền xử lý ảnh theo yêu cầu của ResNet50
        img_array = preprocess_input(img_array)

        # Trích xuất đặc trưng từ mô hình
        feature = model.predict(img_array, verbose=0)

        # Lưu đặc trưng vào danh sách
        features.append(feature)

        # Lưu tên ảnh vào danh sách filenames
        filenames.append(image_path)
    
    return np.vstack(features), filenames

# Đường dẫn chứa ảnh
data_dir = r"C:\Users\Admin\Documents\Python Project\Fashion recommender system (CNN)\Small image\images"

# Khởi tạo mô hình ResNet50
base_model = ResNet50(include_top=False, pooling='avg', input_shape=(224, 224, 3))
base_model.trainable = False  # Không huấn luyện lại ResNet

# Trích xuất đặc trưng và tên ảnh
features, filenames = extract_features_and_filenames(data_dir, base_model)

# In kết quả
print("Shape of extracted features:", features.shape)
print("Number of filenames:", len(filenames))

# Lưu đặc trưng vào file .npy
np.save("features.npy", features)

# Lưu tên ảnh vào file .txt
with open("filenames.txt", "w") as f:
    for filename in filenames:
        f.write(filename + "\n")

print("Features saved to 'features.npy'")
print("Filenames saved to 'filenames.txt'")
