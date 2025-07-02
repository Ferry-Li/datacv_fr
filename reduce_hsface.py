import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from model import iresnet
import torch
from sklearn.cluster import DBSCAN

def load_images_from_dir(dir_path):
    image_paths = []
    for fname in os.listdir(dir_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            image_paths.append(os.path.join(dir_path, fname))
    return sorted(image_paths)

def load_embeddings(image_paths):
    embeddings = []
    valid_image_paths = []

    for path in os.listdir(image_paths):
        if not path.endswith(".npy"):
            continue
        embed_path = os.path.join(image_paths, path)
        idx_name = path.split(".")[0]

        embed = np.load(embed_path)
        embeddings.append(embed)
        valid_image_paths.append(idx_name)

    return embeddings, valid_image_paths

def extract_embeddings(image_paths, fr_model):
   
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    
    embeddings = []
    valid_image_paths = []

    for path in image_paths:
        

        img = cv2.imread(path)  # HWC, BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
        # Normalize
        img = ((img / 255.0) - mean) / std  # shape: HWC
        # Convert to CHW
        img = img.transpose(2, 0, 1)  # shape: CHW

        # Convert to float32 tensor
        img = torch.tensor(img, dtype=torch.float32)

        embeddings.append(fr_model(img))
        valid_image_paths.append(path)

    return embeddings, valid_image_paths

def group_by_identity(embeddings, threshold=0.5):
    n = len(embeddings)
    visited = [False] * n
    groups = []

    for i in tqdm(range(n)):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in range(i + 1, n):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim > threshold and not visited[j]:
                group.append(j)
                visited[j] = True
        groups.append(group)

    return max(groups, key=len) if groups else []

def group_by_identity_dbscan(embeddings, threshold=0.6, min_samples=2):
    """
    Cluster embeddings using DBSCAN and return the largest cluster.
    """
    if isinstance(embeddings[0], torch.Tensor):
        embeddings = [e.detach().cpu().numpy() if e.requires_grad else e.cpu().numpy() for e in embeddings]
    embeddings = np.stack(embeddings)

    # DBSCAN with cosine distance (1 - cosine similarity)
    clustering = DBSCAN(eps=1 - threshold, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings)

    if np.all(labels == -1):  # all noise
        print("No clusters found.")
        return []

    # Find largest cluster label (excluding noise)
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    largest_cluster = unique[np.argmax(counts)]

    # Return indices of samples in the largest cluster
    return [i for i, label in enumerate(labels) if label == largest_cluster]

def group_by_identity_dbscan_auto(
    embeddings,
    initial_threshold=0.6,
    min_samples=2,
    min_percent=0.5,
    max_percent=0.8,
    max_iter=10,
    step=0.02
):
    """
    Dynamically adjust DBSCAN threshold to keep the largest cluster within [min_percent, max_percent].
    """
    if isinstance(embeddings[0], torch.Tensor):
        embeddings = [e.detach().cpu().numpy() if e.requires_grad else e.cpu().numpy() for e in embeddings]
    embeddings = np.stack(embeddings)

    n = len(embeddings)
    threshold = initial_threshold
    best_group = []
    best_percent = 0.0

    for _ in range(max_iter):
        clustering = DBSCAN(eps=1 - threshold, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        if np.all(labels == -1):
            print(f"No clusters found at threshold {threshold:.2f}")
            threshold += step  # try looser threshold
            continue

        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        largest_cluster = unique[np.argmax(counts)]
        group_indices = [i for i, label in enumerate(labels) if label == largest_cluster]
        percent = len(group_indices) / n

        print(f"Threshold: {threshold:.2f}, Cluster Size: {len(group_indices)}, Percent: {percent:.2%}")

        if min_percent <= percent <= max_percent:
            return group_indices  # good enough

        # Save best so far
        if percent > best_percent:
            best_percent = percent
            best_group = group_indices

        # Adjust threshold
        if percent < min_percent:
            threshold -= step
        elif percent > max_percent:
            threshold += step

        # Clamp between [0.3, 0.9] to avoid extreme values
        threshold = max(0.3, min(0.9, threshold))

    print(f"Best cluster found at ~{best_percent:.2%} of total")
    return best_group



def save_filenames(group_indices, valid_image_paths, dir_path):
    save_txt_root = "fr_training/hsface100k/hsface100k"
    filename = dir_path.split("/")[-1] + '.txt'
    output_name = os.path.join(save_txt_root, filename)

    # Get filenames and sort by numeric value if possible
    selected_files = [os.path.basename(valid_image_paths[i]) for i in group_indices]

    # Sort filenames numerically (e.g., "000001.npy" → 1)
    def extract_index(name):
        return int(os.path.splitext(name)[0])

    selected_files.sort(key=extract_index)

    with open(output_name, "w") as f:
        for filename in selected_files:
            f.write(filename + "\n")

    print(f"Saved to {output_name}: {selected_files}")
    print(f"Total files: {len(selected_files)}")


def main(dir_path):
    print(f"Analyzing images in: {dir_path}")
    image_paths = load_images_from_dir(dir_path)
    have_embeddings = True
    if not image_paths:
        print("No images found.")
        return

    fr_model = iresnet(arch="50", fp16=True)

    if not have_embeddings:
        embeddings, valid_image_paths = extract_embeddings(image_paths, fr_model)
        if not embeddings:
            print("No valid face embeddings extracted.")
            return
    else:
        embeddings, valid_image_paths = load_embeddings(dir_path)

    group = group_by_identity_dbscan_auto(embeddings)
    save_filenames(group, valid_image_paths, dir_path)

if __name__ == "__main__":
    root_dir = "fr_training/hsface100k/hsface100k"
    for ID in tqdm(range(0, 100000)):
        dir_path = os.path.join(root_dir, f"{ID:06d}")
        if os.path.exists(dir_path):
            print(f"Processing {dir_path}...")
            main(dir_path)
        else:
            print(f"Directory {dir_path} does not exist, skipping.")
