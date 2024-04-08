import cv2
import jax.numpy as jnp
import numpy as np


def load_image(path: str):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return jnp.array(rgb)


def save_image(path: str, data):
    rgb = np.asarray(data) * 255
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
