import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os
from google.cloud import storage
import functions_framework

# Gloabl model variable
model = None



def download_model_file():
    """Download the model from GCP bucket"""

    from google.cloud import storage

    # Model Bucket details
    BUCKET_NAME = os.environ.get('MODEL_BUCKET')
    PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
    GCS_MODEL_FILE = os.environ.get("MODEL_NAME")

    # Initialise a client
    client = storage.Client(PROJECT_ID)

    # Create a bucket object for our bucket
    bucket = client.get_bucket(BUCKET_NAME)

    # Create a blob object from the filepath
    blob = bucket.blob(GCS_MODEL_FILE)

    folder = '/tmp/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Download the file to a destination
    blob.download_to_filename(folder+'model.h5')


def load_model(path: str):
    """Load model from the tmp directory """

    model = tf.keras.models.load_model(
        path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    model.build((None, 224, 224, 3))
    return model


def process_image(imageNp):
    """Pre-proccess image and convert to numpy array """
    #     Convert to TF
    imageNp = tf.cast(imageNp, tf.float32)
    imageNp = tf.image.resize(imageNp, (224, 224))
    imageNp /= 255
    #     Convert to NP
    imageNp = imageNp.numpy()
    return


def predict(image: Image, model):
    """Predict the date type """

    test_image = np.asarray(image)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)
    probs = model.predict(processed_test_image)
    probs = probs[0].tolist()
    prob, i = tf.math.top_k(probs, k=3)
    probs = prob.numpy().tolist()
    classes = i.numpy().tolist()
    return probs, classes


def download_image(image_path: str):
    """Download image from the bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(os.environ.get(
        "IMAGE_BUCKET"))
    blob = bucket.blob(image_path)
    destinatoin = '/tmp/image.jpg'
    blob.download_to_filename(destinatoin)
    return destinatoin


# @functions_framework.http
def predict_request(request):
    """Main entry point for the serverless function
    1 - get model 
    2 - get image 
    3 - get predction
    """
    global model
    if not model:
        download_model_file()
        model = load_model('tmp/model.h5')

    # Get image from request
    params = request.get_json()
    if (params is not None) and ('img' in params):
        try:

            # get image from storage bucket
            local_image_path = download_image(params['img'])
            img = Image.open(local_image_path)

            # get prediction
            pred_type = predict(model=model, image=img)
            return pred_type, 200
        except Exception as e:
            return f'Error happend while proccessing the prediction Details : {e}'
    else:
        return "couldn't proccess prediction", 400
