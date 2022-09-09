import tensorflow as tf
import tensorflow_hub as hub
import os
from google.cloud import storage
import json

# Gloabl model variable
model = None

img_height = 224
img_width = 224
LABELS_FILE_PATH = 'Model\predict_funciton\label_map.json'
TOP_k = 3


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


def predict(img_path, model):
    """Predict the date type """
    img = tf.keras.utils.load_img(
        img_path, target_size=(img_height, img_width))
    img_path, target_size = (img_height, img_width)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a bat
    probs = model.predict(img_array)
    probs = probs[0].tolist()
    prob, i = tf.math.top_k(probs, k=TOP_k)
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
    4 - transalte prediction
    """
    global model
    if not model:
        download_model_file()
        model = load_model('/tmp/model.h5')

    # Get image from request
    params = request.get_json()
    if (params is not None) and ('img' in params):
        try:
            # get image from storage bucket
            image_bucket_path = params['img']
            local_image_path = download_image(image_bucket_path)
            # get prediction
            probs, classes = predict(model=model, img_path=local_image_path)
            # translate prediction results
            results = {}
            with open(LABELS_FILE_PATH, 'r')as f:
                class_names = json.load(f)
            for i in range(TOP_k):
                results[class_names[str(classes[i])]] = probs[i]
            response = {'img': image_bucket_path, 'results': results}
            return response, 200
        except Exception as e:
            return f'Error happend while proccessing the prediction Details : {e.with_traceback()}'
    else:
        return "couldn't proccess prediction", 400
