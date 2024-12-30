import requests
import time
from PIL import Image
import numpy as np
import cv2
from io import BytesIO  


DIFFUSION_SERVER = 'http://127.0.0.1:10022/'
serialized_data = {
    "param": "just a test!"
}
img_save_path = './tmp/'

try:
    print(f"Sending data to WorldDreamer server...")
    response = requests.post(
        DIFFUSION_SERVER + "dreamer-api/", json=serialized_data)
    print(response)
    if response.status_code == 200 and 'image' in response.headers['Content-Type']:
        combined_image = Image.open(BytesIO(response.content))
        images_array = np.array(combined_image)
        cv2.imwrite(f"{img_save_path}diffusion_{time.time()}.jpg",
                    cv2.cvtColor(images_array, cv2.COLOR_RGB2BGR))
        
except requests.exceptions.RequestException as e:
    print(f"Warning: Request failed due to {e}")