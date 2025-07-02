import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def low_resolution(img, scale_factor=0.5):
    w, h = img.size
    downsampled = img.resize((int(w * scale_factor), int(h * scale_factor)), Image.BILINEAR)
    return downsampled.resize((w, h), Image.BILINEAR)

def get_augmentation_pipeline():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5
            )
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=5)
        ], p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Lambda(lambda img: low_resolution(img, scale_factor=0.5)),
    ])

def cleanup_and_augment(dir_path, target_count=50, log_file="too_few_samples.txt"):
    # txt_path = os.path.join(dir_path, os.path.basename(dir_path) + ".txt")
    id_name = dir_path.split("/")[-1]
    txt_path = os.path.join("fr_training/hsface10k_gpt4o/hsface10k/", f"{id_name}.txt")
    if not os.path.exists(txt_path):
        print(f"ID list file not found: {txt_path}")
        return

    # Step 1: Read allowed image IDs from .txt
    with open(txt_path, "r") as f:
        allowed_ids = set(line.strip().split('.')[0] for line in f if line.strip())

    # Step 2: Log if fewer than 10
    if len(allowed_ids) < 10:
        with open(log_file, "a") as log:
            log.write(os.path.basename(id_name) + "\n")
        print(f"{dir_path}: Only {len(allowed_ids)} images — logged to {log_file}")

    if len(allowed_ids) < 1:
        print(f"No valid IDs found in {txt_path}, skipping {dir_path}.")
        return

    # Step 3: Remove images not in the allowed list
    image_files = sorted([
        f for f in os.listdir(dir_path)
        if f.lower().endswith('.jpg') and os.path.isfile(os.path.join(dir_path, f))
    ])

    kept_images = []
    for f in image_files:
        id_str = os.path.splitext(f)[0]
        if id_str not in allowed_ids:
            os.remove(os.path.join(dir_path, f))
        else:
            kept_images.append(f)

    print(f"Cleaned up. Kept {len(kept_images)} images.")

    # Step 4: Augment if needed
    if len(kept_images) >= target_count:
        print(f"Already has {len(kept_images)} images (>= {target_count}), skipping augmentation.")
        return

    aug = get_augmentation_pipeline()
    needed = target_count - len(kept_images)
    print(f"Augmenting {needed} more images...")

    base_images = [
        Image.open(os.path.join(dir_path, f)).convert("RGB")
        for f in kept_images
    ]

    used_ids = set(int(os.path.splitext(f)[0]) for f in kept_images)
    next_id = max(used_ids) + 1 if used_ids else 0

    for _ in range(needed):
        base_img = random.choice(base_images)
        aug_img = aug(base_img)
        filename = f"{next_id:03d}.jpg"
        aug_img.save(os.path.join(dir_path, filename))
        next_id += 1

    print(f"Final total: {target_count} images in {dir_path}")

# Example usage
if __name__ == "__main__":
    root_path = "fr_training/hsface10k_gpt4o/hsface10k"

    for ID in tqdm(range(0, 10000)):
        dir_path = os.path.join(root_path, f"{ID:06d}")
        if os.path.exists(dir_path):
            print(f"Processing {dir_path}...")
            cleanup_and_augment(dir_path, target_count=50, log_file="hsface10K_gpt4o_too_few_examples.txt")
        else:
            print(f"Directory {dir_path} does not exist, skipping.")