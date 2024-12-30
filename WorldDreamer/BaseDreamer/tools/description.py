import base64
import concurrent.futures
import copy
import pickle
import time
from functools import partial

import requests


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_openai_description(image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)
    api_key = ""
    proxies = {
        "http": "",
        "https": "",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
            Based on the driving image, you need to give the following CORE information about it:
            - Time of the day: daytime or night.
            - Weather: Sunny, rainy, cloudy, or snowy
            - Surrounding environment: downtown, suburban, rural, or nature.
            - Road condition: (The car is driving on) intersection/straight road/narrow street/wide road/ped crossing/ etc. 
            - Give 2-3 key words to desctibe other key infomation about surrounding, especially colors of the buildings/vehicles/trees.
            Your answer should be several keywords seperate by commas, no need for a sentence. e.g "daytime, cloudy, nature, wide street, white building, green trees."
            """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 100,
    }
    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                proxies=proxies,
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, requests.RequestException) as e:
            if attempt < max_retries - 1:
                print(f"Error occurred: {e}. Retrying in 1 seconds...")
                time.sleep(1)
            else:
                print(f"Max retries reached. Last error: {e}")
                return "Failed to get description after multiple attempts."
    return description


def process_scene(scene, token_data_dict, data_infos, get_openai_description):
    idx = int(len(scene) / 2)
    image_path = data_infos[token_data_dict[scene[idx]]]["cams"]["CAM_F0"]["data_path"]
    description = get_openai_description(image_path)
    print(description)
    return scene, description


file = open(
    "data/nuplan/nuplan_infos_val.pkl",
    "rb",
)
datas = pickle.load(file)
data_infos = list(sorted(datas["infos"], key=lambda e: e["timestamp"]))
data_infos = data_infos[::1]
scene_tokens = datas["scene_tokens"]
token_data_dict = {item["token"]: idx for idx, item in enumerate(data_infos)}
print(f'Available scenes: {len(scene_tokens)}')

with concurrent.futures.ThreadPoolExecutor(
    max_workers=1
) as executor:  
    process_scene_partial = partial(
        process_scene,
        token_data_dict=token_data_dict,
        data_infos=data_infos,
        get_openai_description=get_openai_description,
    )
    results = list(executor.map(process_scene_partial, scene_tokens))
for i, (scene, description) in enumerate(results):
    print(f"scene {i}", description)
    for token in scene:
        data_infos[token_data_dict[token]]["description"] = description

with open(
    "data/nuplan/nuplan_infos_val_with_note.pkl",
    "wb",
) as f:
    data_copy = copy.deepcopy(datas)
    data_copy["infos"] = data_infos
    pickle.dump(data_copy, f)
