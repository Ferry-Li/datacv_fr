import os
import json

# Paths
json_path = "gpt4o_clean_results_cleaned.json"  # path to your JSON file
image_root = "fr_training/hsface10k_gpt4o/hsface10k"  # root directory with ID_xxxxx folders
output_txt_dir = "hsface10k_gpt4o/hsface10k"  # where to save the .txt files
os.makedirs(output_txt_dir, exist_ok=True)

# Load JSON
with open(json_path, 'r') as f:
    to_remove_dict = json.load(f)

# Process each ID
for id_key, remove_str in to_remove_dict.items():
    id_dir = os.path.join(image_root, f"{id_key}")
    if not os.path.exists(id_dir):
        print(f"Directory not found: {id_dir}")
        continue

    # List all .jpg image names (without .jpg extension)
    all_images = [
        os.path.splitext(f)[0]
        for f in os.listdir(id_dir)
        if f.lower().endswith(".jpg")
    ]

    # Set of images to remove
    remove_set = set(remove_str.split(","))

    # Preserve = all - to_remove
    preserve_images = sorted(set(all_images) - remove_set, key=lambda x: int(x))

    # Write to txt
    txt_path = os.path.join(output_txt_dir, f"{id_key}.txt")
    with open(txt_path, "w") as f:
        for img in preserve_images:
            f.write(f"{img}\n")

    print(f"Saved: {txt_path} ({len(preserve_images)} preserved)")