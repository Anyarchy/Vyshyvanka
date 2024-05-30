import os
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import json
from torchvision import models
from torch import nn
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/Dataset/'
app.config['REGION_IMAGES'] = 'static/image/regions'

with open('static/region_names.json', encoding='utf-8') as json_file:
    regions = json.load(json_file)

# Завантаження індексів класів
with open('static/model/class_indices.json') as json_file:
    class_indices = json.load(json_file)

# Визначення моделі
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Кількість класів
num_classes = len(class_indices)
model = ResNetClassifier(num_classes)

# Завантаження ваг моделі
model_path = 'static/model/vyshyvanka_model_weights.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Трансформації для зображень
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    class_folders = [d for d in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], d))]
    images = {}
    for class_name in class_folders:
        class_path = os.path.join(app.config['UPLOAD_FOLDER'], class_name)
        images[class_name] = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]

    region_images_folder = app.config['REGION_IMAGES']
    region_images = [os.path.splitext(img)[0] for img in os.listdir(region_images_folder) if os.path.isfile(os.path.join(region_images_folder, img))]

    return render_template('home.html', classes=class_folders, images=images, region_images=region_images, regions=regions)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('predict', filename=filename))
    return redirect(url_for('home'))

def predict_and_render(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(filepath)
    image = transform(image).unsqueeze(0)  # Додавання batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class_index = predicted.item()
        predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
        predicted_confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_class_index].item()

    similar_images = get_similar_images(predicted_class_name)
    return render_template('result_modal_content.html', class_name=predicted_class_name, confidence=predicted_confidence, filename=filename, similar_images=similar_images)


@app.route('/predict/<filename>')
def predict(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(filepath)
    image = transform(image).unsqueeze(0)  # Додавання batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_classes = torch.topk(probabilities, k=3)

    results = []
    for i in range(len(top_probs)):
        class_index = top_classes[i].item()
        class_name = class_indices.get(str(class_index), "Unknown")
        confidence = top_probs[i].item() * 100  # Відсотки
        results.append({'class_name': class_name, 'confidence': confidence})

    # Отримати схожі зображення для регіону з найбільшою впевненістю
    most_confident_class = results[0]['class_name']
    similar_images = get_similar_images(most_confident_class)

    class_folder = os.path.join(app.config['UPLOAD_FOLDER'], most_confident_class)
    all_images = [image for image in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, image))]

    return render_template('result_modal_content.html', results=results, filename=filename, similar_images=similar_images, all_images=all_images)


def get_similar_images(class_name):
    class_folder = os.path.join(app.config['UPLOAD_FOLDER'], class_name)
    images = [image for image in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, image))]
    return images[:3]

@app.route('/correct/<filename>')
def correct(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    test_image = load_and_prepare_image(filepath)
    prediction = model(test_image)
    predicted_class_index = torch.argmax(prediction, axis=1).item()
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    new_folder = os.path.join(app.config['UPLOAD_FOLDER'], predicted_class_name)

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    safe_file_move(filepath, new_folder, filename)
    return redirect(url_for('home'))

@app.route('/incorrect/<filename>', methods=['GET', 'POST'])
def incorrect(filename):
    if request.method == 'POST':
        new_class = request.form['new_class']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        new_folder = os.path.join(app.config['UPLOAD_FOLDER'], new_class)

        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        safe_file_move(filepath, new_folder, filename)
        return redirect(url_for('home'))
    return render_template('incorrect.html', filename=filename, classes=class_indices.values())

@app.route('/unknown/<filename>')
def unknown(filename):
    return redirect(url_for('home'))

def load_and_prepare_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def safe_file_move(filepath, target_folder, filename):
    import uuid
    new_filepath = os.path.join(target_folder, filename)
    if os.path.exists(new_filepath):
        unique_id = uuid.uuid4().hex
        name, extension = os.path.splitext(filename)
        new_filename = f"{name}_{unique_id}{extension}"
        new_filepath = os.path.join(target_folder, new_filename)

    os.rename(filepath, new_filepath)
    return new_filepath

if __name__ == '__main__':
    app.run(debug=True)