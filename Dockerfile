FROM python:3.7-slim

COPY requirements.txt /hate_meme_app/requirements.txt
WORKDIR /hate_meme_app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

#RUN ["python", "./utils/download_trained_model.py"]

EXPOSE 5000
 
CMD [ "python", "./app.py" ]