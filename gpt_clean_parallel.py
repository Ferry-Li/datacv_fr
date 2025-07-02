import os
import json
import re
import time
import base64
from PIL import Image
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Explicitly pass API Key
client = OpenAI(api_key="API_KEY", base_url="https://api.chatanywhere.tech/v1")

# Image directory and output result path
image_dir = 'hsface20k'
output_json = 'gpt4o_clean_results.json'

# Check if response is valid (e.g., "001,005,012")
def is_valid_response(response):
    return re.fullmatch(r"(\d{3})(,\d{3})*", response.strip()) is not None

# Send image and prompt to GPT
def ask_identity_check(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You will receive an image consisting of several face photos arranged in a grid, where each face has a numeric ID label below it. Please identify all the face images that do not belong to the same person as the majority. Your answer should only include the numeric IDs of the outlier faces, separated by commas (e.g., 001,005,012). Do not include any additional text, punctuation, or whitespace. Do not return an empty response."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    for _ in range(3):  # Retry up to 3 times
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=50,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(1)
    return None

# Wrapper for processing a single image (used in parallel)
def process_one(image_file):
    image_path = os.path.join(image_dir, image_file)
    key = os.path.splitext(image_file)[0]
    answer = ask_identity_check(image_path)
    return key, answer

# Load existing results (supports resuming)
if os.path.exists(output_json):
    with open(output_json, 'r') as f:
        results = json.load(f)
else:
    results = {}

# Prepare list of unprocessed images
image_files = sorted(f for f in os.listdir(image_dir) if f.endswith('.png'))
pending_files = [f for f in image_files if os.path.splitext(f)[0] not in results]

# Set max number of concurrent threads (e.g., 20)
max_workers = 20

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_file = {executor.submit(process_one, image_file): image_file for image_file in pending_files}

    for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing"):
        image_file = future_to_file[future]
        key = os.path.splitext(image_file)[0]
        try:
            result_key, answer = future.result()
            if answer:
                results[result_key] = answer
                print(f"[√] {image_file} -> {answer}")
            else:
                print(f"[x] {image_file} -> Failed to get response.")
        except Exception as e:
            print(f"[!] Exception processing {image_file}: {e}")

        # Save results in real-time
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)