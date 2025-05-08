import tensorflow as tf, numpy as np, random
import tensorflow_hub as hub
from PIL import Image

def stylize(content_path, style_path, max_dim=512):
    def proc(p):
        img = Image.open(p).convert('RGB')
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
        t = tf.image.convert_image_dtype(np.array(img), tf.float32)[None]
        return t

    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    out = model(proc(content_path), proc(style_path))[0][0]
    out = (out*255).numpy().astype('uint8')
    Image.fromarray(out).show()

# Usage:
stylize('base.jpg', 'style.jpg')