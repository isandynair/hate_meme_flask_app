import gdown
import os

url = 'https://drive.google.com/uc?id=1DVzv2LUgKeT9ZmQSdDwQ2SmoVK8N6ra8'

if not 'model' in os.listdir('.'):
    os.makedirs('model', exist_ok=True)

output = './model/meme_model.pt'
gdown.download(url, output, quiet=False)

print('Downloading trained model completed!!')