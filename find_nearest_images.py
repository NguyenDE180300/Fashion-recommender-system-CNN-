import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

small_images_dir = r"C:\Users\Admin\Documents\Python Project\Fashion recommender system (CNN)\Small image\images"
large_images_dir = r"C:\Users\Admin\Documents\Python Project\Fashion recommender system (CNN)\Original image\fashion-dataset\images"

small_filenames = [os.path.join(small_images_dir, fname) for fname in os.listdir(small_images_dir)]

filename_map = {os.path.basename(fname): os.path.join(large_images_dir, os.path.basename(fname)) for fname in small_filenames}

features = np.load("features.npy")

def find_similar_images(selected_feature, all_features, filenames, top_n=5):
    similarities = cosine_similarity(selected_feature.reshape(1, -1), all_features)
    sorted_indices = similarities.argsort()[0][::-1]

    similar_images = []``
    for idx in sorted_indices:
        if similarities[0][idx] < 1.0:
            similar_images.append((filenames[idx], similarities[0][idx]))
        if len(similar_images) == top_n:
            break

    return similar_images

def extract_feature_from_image(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array, verbose=0)
    return feature

def display_images(image_paths, similarities=None, title="Images"):
    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(image_paths):
        img = mpimg.imread(img_path)
        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(img)
        plt.axis('off')
        if similarities is not None:
            plt.title(f"Sim: {similarities[i]:.2f}")
    plt.suptitle(title)
    plt.show()

new_image_path = r"C:\Users\Admin\Documents\Python Project\Fashion recommender system (CNN)\download.jpg"  # Đường dẫn ảnh của bạn

base_model = ResNet50(include_top=False, pooling='avg', input_shape=(224, 224, 3))
base_model.trainable = False

new_image_feature = extract_feature_from_image(new_image_path, base_model)

similar_images = find_similar_images(new_image_feature, features, small_filenames, top_n=5)

print("Selected image:", new_image_path)
original_image = [new_image_path]
display_images(original_image, title="Original Image")

print("Most similar images:")
similar_image_paths = [filename_map[os.path.basename(img)] for img, _ in similar_images]
similarities = [sim for _, sim in similar_images]
display_images(similar_image_paths, similarities, title="Similar Images")
