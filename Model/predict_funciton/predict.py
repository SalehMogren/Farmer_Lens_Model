from msilib.schema import Error
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os
from google.cloud import storage
import functions_framework


## Gloabl model variable
model = None

# Download model file from cloud storage bucket
def download_model_file():

    from google.cloud import storage

    # Model Bucket details
    BUCKET_NAME        = os.environ['MODEL_BUCKET']
    PROJECT_ID         = os.environ["GCP_PROJECT_ID"]
    GCS_MODEL_FILE     = os.environ["MODEL_NAME"]

    # Initialise a client
    client   = storage.Client(PROJECT_ID)
    
    # Create a bucket object for our bucket
    bucket   = client.get_bucket(BUCKET_NAME)
    
    # Create a blob object from the filepath
    blob     = bucket.blob(GCS_MODEL_FILE)
    
    folder = '/tmp/'
    if not os.path.exists(folder):
      os.makedirs(folder)
    # Download the file to a destination
    blob.download_to_filename(folder + GCS_MODEL_FILE)

def load_model(path:str):
    model = tf.keras.models.load_model(path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    model.build((None, 224, 224, 3))
    return model

def process_image(imageNp):
    #     Convert to TF
    imageNp = tf.cast(imageNp, tf.float32)
    imageNp = tf.image.resize(imageNp, (224, 224))
    imageNp /= 255
    #     Convert to NP
    imageNp = imageNp.numpy()
    return 

def predict(image:Image,model):
    test_image = np.asarray(image)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)
    probs = model.predict(processed_test_image)
    probs = probs[0].tolist()
    prob, i = tf.math.top_k(probs, k=3)
    probs = prob.numpy().tolist()
    classes = i.numpy().tolist()
    return probs, classes

def download_image(image_path:str):

    """Downloads a blob from the bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(os.environ["IMAGE_BUCKET"])
    blob = bucket.blob(image_path)
    destinatoin = '/tmp/image.jpg'
    blob.download_to_filename(destinatoin)
    return destinatoin

@functions_framework.http
def predict_request(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    global model
    if not model:
        download_model_file()
        model = load_model(f'/tmp/{os.environ["MODEL_NAME"]}')

    # Get image from request 
    params = request.get_json()
    if (params is not None) and ('img' in params):
        try:
                
            # get image from storage bucket
            local_image_path = download_image(params['img'])
            img = Image.open(local_image_path)

            # get prediction
            pred_type = predict(model=model,image=img)
            return pred_type,200
        except Exception as e:
            return f'Error happend while proccessing the prediction Details : {e}'
    else:
        return "couldn't proccess prediction",400

