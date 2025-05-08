import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Load the style transfer model from TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to load and preprocess image
def load_image(image_url, img_size=(512, 512)):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize(img_size, Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0
    img = img[tf.newaxis, ...]
    return tf.cast(img, tf.float32)

# URLs for content and style images
content_image_url = 'https://raw.githubusercontent.com/tensorflow/docs/master/site/en/hub/tutorials/images/dog.jpg'
style_image_url = 'https://raw.githubusercontent.com/tensorflow/docs/master/site/en/hub/tutorials/images/starry_night.jpg'

# Load content and style images
content_image = load_image(content_image_url)
style_image = load_image(style_image_url)

# Perform neural style transfer
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Convert the stylized image to a displayable format
stylized_image = np.clip(stylized_image[0].numpy(), 0, 1)

# Save and display the result
plt.figure(figsize=(10, 10))
plt.imshow(stylized_image)
plt.axis('off')
plt.title('Stylized Image')
plt.savefig('stylized_image.png')
plt.close()