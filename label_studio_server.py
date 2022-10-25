import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
import requests
import io
import hashlib
import urllib
import boto3
import pickle

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_choice, is_skipped, get_local_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


image_size = 224
image_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_cache_dir = os.path.join(os.path.dirname(__file__), 'image-cache')
os.makedirs(image_cache_dir, exist_ok=True)


def load_image(url):
    fields = url.split("?")[0].split("/")
    bucket = fields[2]
    key = "/".join(fields[3:])
    endpoint_url = os.environ["S3_ENDPOINT"]
    s3 = boto3.client('s3',endpoint_url='https://minio.lab.sspcloud.fr/')
    data = s3.get_object(Bucket=bucket, Key=key)
    img = Image.open(data["Body"]).convert("RGB")

    return image_transforms(img)

def import_model(model, path="mbenxsalha/diffusion/state_dict.pickle"):
    if path.endswith("pickle"):
        bucket = path.split("/")[0]
        key = '/'.join(path.split("/")[1:])
        endpoint_url = os.environ["S3_ENDPOINT"]
        s3 = boto3.client('s3',endpoint_url="https://minio.lab.sspcloud.fr")
        data = s3.get_object(Bucket=bucket, Key=key)
        state_dict = pickle.loads(data['Body'].read())
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(path))
    model.eval()
    return model


class ImageClassifierDataset(Dataset):

    def __init__(self, image_urls, image_classes):
        self.classes = list(set(image_classes))
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        self.images, self.labels = [], []
        for image_url, image_class in zip(image_urls, image_classes):
            try:
                image = load_image(image_url)
            except Exception as exc:
                print(exc)
                continue
            self.images.append(image)
            self.labels.append(self.class_to_label[image_class])

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

class ResNet(nn.Module):
    def __init__(self, class_num=2, pretrained=True):
        super(ResNet, self).__init__()
        model = models.resnet18(pretrained=pretrained)
        fc_input_dim = model.fc.in_features
        model.fc = nn.Linear(fc_input_dim, class_num)
        self.model = model
    def forward(self, x):
        x = self.model(x)
        return x
        
    
class ImageClassifier(object):

    def __init__(self, class_num=2, path="mbenxsalha/diffusion/state_dict.pickle"):  
        self.model = ResNet(class_num)
        if path:
            self.model = import_model(self.model, path)
        self.model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
   
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model = import_model(self.model, path)
        
    def predict(self, image_urls):
        imgs = torch.stack([load_image(image_url) for image_url in image_urls], dim=0)
        imgs = imgs.to(device)
        print(imgs.shape)
        with torch.no_grad():
            preds = self.model(imgs)
        return preds.to(device).data.numpy()

    def train(self, dataloader, num_epochs=5, lr=0.001, momentum=0.9, weight_decay=0.0001):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        return self.model

    


class ImageClassifierAPI(LabelStudioMLBase):

    def __init__(self, freeze_extractor=False, **kwargs):
        super(ImageClassifierAPI, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'Choices', 'Image')
        if self.train_output:
            self.model = ImageClassifier(len(self.classes))
            self.model.load(self.train_output['model_path'])
        else:
            self.model = ImageClassifier(len(self.classes))
        print(self.classes)
    def reset_model(self):
        self.model = ImageClassifier(len(self.classes))

    def predict(self, tasks, **kwargs):
        image_urls = [task['data'][self.value] for task in tasks]
        logits = self.model.predict(image_urls)
        predicted_label_indices = np.argmax(logits, axis=1)
        predicted_scores = logits[np.arange(len(predicted_label_indices)), predicted_label_indices]
        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.classes[idx]
            # prediction result for the single task
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            # expand predictions with their scores for all tasks
            predictions.append({'result': result, 'score': float(score)})

        return predictions

    def fit(self, completions, workdir=None, batch_size=32, num_epochs=10, **kwargs):
        image_urls, image_classes = [], []
        print('Collecting annotations')
        for completion in completions:
            if is_skipped(completion):
                continue
            image_urls.append(completion['data'][self.value])
            image_classes.append(get_choice(completion))

        print(f'Creating dataset with {len(image_urls)} images')
        dataset = ImageClassifierDataset(image_urls, image_classes)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        print('Train model')
        self.reset_model()
        self.model.train(dataloader, num_epochs=num_epochs)

        print('Save model')
        model_path = os.path.join(workdir, 'model.pth')
        self.model.save(model_path)

        return {'model_path': model_path, 'classes': dataset.classes}
