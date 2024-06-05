from django.shortcuts import render
from .forms import DeploymentForm
from django.http import HttpResponse
from PIL import Image
import pandas as pd
import os
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tensorflow import keras
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_int)

# Create your views here.
def home(request):
    return render(request, 'index.html')

def start(request):
    form = DeploymentForm()
    if request.method == 'POST':
        form = DeploymentForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            file = request.FILES['File_To_Upload']
            u = predict(file)
            image_path = "C:/DJANGOO/_PBL/myapp/static/img/new2.png"
            u.save(image_path)
            return render(request, 'result.html', {'image_path': image_path})
    context = {'form': form}
    return render(request, 'start.html', context)

# Load the EEG classifier
eeg_classifier = keras.models.load_model('C:/Users/ahmed/Desktop/ThoughtViz_obj_eeg.h5')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the discriminator
class Discriminator(nn.Module):
    # Define the discriminator architecture
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load('C:/Users/ahmed/Desktop/discriminator.pth'), strict=False)
discriminator.eval()

# Load pre-trained BigGAN generator
generator = BigGAN.from_pretrained('biggan-deep-128')
generator.to(device)
for param in generator.parameters():
    param.requires_grad = False

# Define the mapping between EEG classes and BigGAN classes
eeg_to_biggan_mapping = {
    0: 957, 1: 817, 2: 239, 3: 705, 4: 487,
    5: 989, 6: 670, 7: 292, 8: 893, 9: 826
}



def predict(file):
    eeg_data = pickle.load(file, encoding='latin1')
    x_test = eeg_data['x_test']
    # Use the first data point for prediction
    data_point = x_test[57].reshape(1, 14, 32, 1)

    # Predict EEG class
    eeg_class_label = np.argmax(eeg_classifier.predict(data_point), axis=1)[0]
    biggan_class_label = eeg_to_biggan_mapping[eeg_class_label]

    # Generate BigGAN image
    noise = truncated_noise_sample(truncation=0.4, batch_size=1)
    noise = torch.from_numpy(noise).to(device)
    class_label_onehot = one_hot_from_int([biggan_class_label], batch_size=1)
    class_label_onehot = torch.from_numpy(class_label_onehot).to(device)

    with torch.no_grad():
        fake_image = generator(noise, class_label_onehot, truncation=0.4)

    # Convert the image to PIL format
    fake_image_pil = Image.fromarray(((fake_image.cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5).astype(np.uint8)[0])

    return fake_image_pil
