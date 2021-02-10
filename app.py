import json
import logging
from functools import lru_cache
from pathlib import Path
import random
import tarfile
import tempfile
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sentence_transformers import SentenceTransformer
from PIL import Image
import torch                    
import torchvision
import pytorch_lightning as pl

from utils.hparams import hparams
from utils.download_trained_model import download_model

from flask import Flask, render_template, request, flash, redirect
#from flask_uploads import UploadSet, configure_uploads,IMAGES


try:
    from utils.model import HatefulMemesModel
except:
    import sys
    sys.path.append("utils")
    from utils.model import HatefulMemesModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


app = Flask(__name__)
UPLOAD_FOLDER = "static"

download_model("https://drive.google.com/uc?id=1DVzv2LUgKeT9ZmQSdDwQ2SmoVK8N6ra8")


def load_model(model_path):

    '''Loads the trained model from model folder'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    concat_meme_model = HatefulMemesModel(hparams)
    concat_meme_model.load_state_dict(checkpoint['state_dict'])

    #checkpt = torch.load(model_path, map_location=device)
    #concat_meme_model = HatefulMemesModel(hparams)
    #concat_meme_model.load_state_dict(checkpt['state_dict'])
    
    return concat_meme_model

def load_callable_encoders():

    '''Prepares text and image feature callable encoders'''

    image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    size=(224, 224)
                ),        
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    text_transform = SentenceTransformer(
        'distilbert-base-nli-stsb-mean-tokens'
        )
    return text_transform, image_transform

def predict_hatefulness(txt:str, img, model, text_transform, image_transform):

    ''' Predicts hatefulness of meme image '''

    text = torch.Tensor(text_transform.encode(txt)).squeeze().unsqueeze(0)
    image = image_transform(Image.open(img).convert("RGB")).unsqueeze(0)
    preds, _ = model.eval().to("cpu")(text, image)

    if preds.argmax(dim=1) == 1:
        prediction = "Hateful"
    else:
        prediction = "Non-Hateful"

    proba = round(preds[:, 1].item(), 4)*100

    return prediction, proba

@app.route("/", methods=["GET", "POST"])
def landing_page():
    if request.method == "POST":
        if "text_input" not in request.form or "meme_image" not in request.files:
            return redirect(request.url)
        

        text = request.form["text_input"]
        image = request.files["meme_image"]

        if text and image:

            prediction, proba = predict_hatefulness(txt=text,
                                                    img=image,
                                                    model=concat_meme_model,
                                                    text_transform=text_transform,
                                                    image_transform=image_transform)

            return render_template("index.html", meme_tox=prediction, probability=proba)
        
        
    return render_template("index.html")

concat_meme_model = load_model(model_path = "./model/meme_model.pt")
text_transform, image_transform = load_callable_encoders()


if __name__ == '__main__':

    app.run(port=5000, debug=True, host="0.0.0.0")