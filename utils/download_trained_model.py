import gdown
import os

def download_model(path):

    url = path

    if not 'model' in os.listdir('.'):
        os.makedirs('model', exist_ok=True)

        output = './model/meme_model.pt'
        gdown.download(url, output, quiet=False)

    return 'Downloading trained model completed!!'