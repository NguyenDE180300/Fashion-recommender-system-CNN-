from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# Đọc file styles.csv
styles_df = pd.read_csv(r"C:\Users\Admin\Documents\Python Project\Fashion recommender system (CNN)\Small image\styles.csv", usecols=['id', 'subCategory'])

# Tạo ánh xạ giữa id và subCategory
id_to_category = styles_df.set_index('id')['subCategory'].to_dict()


# Cấu hình Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploaded'

# Load model ResNet50
model = ResNet50(include_top=False, pooling='avg', input_shape=(224, 224, 3))
model.trainable = False

# Load features.npy và filename_map
features = np.load("features.npy")
small_images_dir = r"C:\Users\Admin\Documents\Python Project\Fashion recommender system (CNN)\Small image\images"
large_images_dir = r"C:\Users\Admin\Documents\Python Project\Fashion recommender system (CNN)\Original image\fashion-dataset\images"
small_filenames = [os.path.join(small_images_dir, fname) for fname in os.listdir(small_images_dir)]
filename_map = {os.path.basename(fname): os.path.join(large_images_dir, os.path.basename(fname)) for fname in small_filenames}

# Hàm trích xuất đặc trưng
def extract_feature_from_image(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array, verbose=0)
    return feature

categories = [
    'Accessories', 'Apparel Set', 'Bag', 'Bath and Body', 'Belts', 'Beauty Accessories', 'Bottomwear', 
    'Cufflinks', 'Dress', 'Eyewear', 'Flip Flops', 'Fragrance', 'Free Gifts', 'Gloves', 'Hair', 'Headwear',
    'Innerwear', 'Jewellery', 'Lips', 'Loungewear and Nightwear', 'Makeup', 'Mufflers', 'Nails', 'Perfumes',
    'Sandal', 'Saree', 'Scarves', 'Shoe Accessories', 'Shoes', 'Skin', 'Skin Care', 'Socks', 'Sports Accessories', 
    'Sports Equipment', 'Stoles', 'Ties', 'Topwear', 'Umbrellas', 'Vouchers', 'Water Bottle', 'Watches', 'Wallets', 
    'Wristbands'
]

# Hàm tìm kiếm hình ảnh tương tự
def find_similar_images(selected_feature, all_features, filenames, top_n=5, selected_category=None):
    similarities = cosine_similarity(selected_feature.reshape(1, -1), all_features)
    sorted_indices = similarities.argsort()[0][::-1]

    similar_images = []
    for idx in sorted_indices:
        # Lấy id từ tên file
        image_id = int(os.path.basename(filenames[idx]).split('.')[0])

        # Kiểm tra nếu ảnh thuộc cùng category (nếu được chỉ định)
        if selected_category and id_to_category.get(image_id) != selected_category:
            continue

        # Thêm ảnh nếu độ tương đồng nhỏ hơn 1.0
        if similarities[0][idx] < 1.0:
            similar_images.append((filenames[idx], similarities[0][idx]))
        if len(similar_images) == top_n:
            break

    return similar_images



# Route phục vụ ảnh từ thư mục khác
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(large_images_dir, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra và lưu ảnh tải lên
        if 'file' not in request.files:
            return "No file uploaded!", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected!", 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Lấy lựa chọn danh mục từ form
        selected_category = request.form.get('category', 'All')

        # Trích xuất đặc trưng từ ảnh tải lên
        feature = extract_feature_from_image(file_path, model)

        # Tìm kiếm ảnh tương tự với danh mục đã chọn
        similar_images = find_similar_images(
            feature, 
            features, 
            small_filenames, 
            top_n=5, 
            selected_category=None if selected_category == 'All' else selected_category
        )
        
        similar_image_paths = [os.path.basename(filename_map[os.path.basename(img)]) for img, _ in similar_images]
        similarities = [sim for _, sim in similar_images]

        # Trả kết quả
        return render_template(
            'index.html', 
            uploaded_image=file_path, 
            similar_images=zip(similar_image_paths, similarities),
            categories=categories
        )

    return render_template('index.html', categories=categories)



if __name__ == '__main__':
    app.run(debug=True)