# Hate Meme Flask App

#### Hate Meme Classifier trained on Vision + Language concatenation model. Most of the model development was inspired by referring this awesome [blog](https://www.drivendata.co/blog/hateful-memes-benchmark/).

### Steps to run:

#### Step 1: Clone repo
```
git clone https://github.com/sandycancode/hate_meme_flask_app.git
cd hate_meme_flask_app
```
#### Step 2: Create virtual environment
```
pip install --upgrade pip
pip install virtualenv
virtualenv flask_env --python=python3.7
source flask_env/bin/activate
```
#### Step 3: Install dependencies
```
pip install -r requirements.txt
```
#### Step 4: Run the flask app
```
flask run
```
##### ***Note: Running `flask run` will automatically download the trained model from [this](https://drive.google.com/file/d/1DVzv2LUgKeT9ZmQSdDwQ2SmoVK8N6ra8/view?usp=sharing) link to newly created `./model` directory!***

<br />

# Azure Deployment

#### The model has also been deployed to Azure via the Azure App Service referring these [steps](https://docs.microsoft.com/en-us/azure/developer/python/tutorial-deploy-app-service-on-linux-01).
#### You can view the app here - https://hatememewebapp.azurewebsites.net/
