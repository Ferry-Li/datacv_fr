import os
import random
from collections import defaultdict


def load_txt(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def extract_id_original(image_path):
    # Extract ID like 000466 from: /.../hsface10k/000466/008.jpg
    return os.path.basename(os.path.dirname(image_path))


def extract_id_synthetic(image_path):
    # Extract ID like ID_05669 from: /.../generated_images_ref/ID_05669/003.jpg
    return os.path.basename(os.path.dirname(image_path))


def create_mixed_dataset(
    original_paths_txt,
    too_few_ids_txt,
    synthetic_paths_txt,
    output_txt
):
    # Load data
    original_paths = load_txt(original_paths_txt)
    too_few_ids = set(load_txt(too_few_ids_txt))
    synthetic_paths = load_txt(synthetic_paths_txt)

    print(f"Original images loaded: {len(original_paths)}")
    print(f"Too-few IDs to remove: {len(too_few_ids)}")
    print(f"Synthetic images loaded: {len(synthetic_paths)}")

    # Step 1: Filter out original paths with too-few IDs
    filtered_original = [
        path for path in original_paths
        if extract_id_original(path) not in too_few_ids
    ]
    num_removed_images = len(original_paths) - len(filtered_original)
    print(f"Remaining original images: {len(filtered_original)} ({num_removed_images} removed)")

    # Step 2: Group synthetic paths by ID
    id_to_synth_paths = defaultdict(list)
    for path in synthetic_paths:
        synth_id = extract_id_synthetic(path)
        id_to_synth_paths[synth_id].append(path)

    # Step 3: For each removed ID, randomly pick one synthetic ID
    removed_ids_count = len(too_few_ids)
    available_synthetic_ids = list(id_to_synth_paths.keys())
    random.shuffle(available_synthetic_ids)

    selected_synthetic_paths = []
    used_ids = set()

    for _ in range(removed_ids_count):
        if not available_synthetic_ids:
            print("Not enough synthetic IDs to fully make up for removed IDs.")
            break
        synth_id = available_synthetic_ids.pop()
        used_ids.add(synth_id)
        selected_synthetic_paths.extend(id_to_synth_paths[synth_id])

    print(f"Used {len(used_ids)} synthetic IDs")
    print(f"Total synthetic images added: {len(selected_synthetic_paths)}")

    # Step 4: Mix synthetic and original (preserve grouping)
    final_list = selected_synthetic_paths + filtered_original

    # Step 5: Save to output
    with open(output_txt, "w") as f:
        for line in final_list:
            f.write(line + "\n")

    print(f"Final dataset written to: {output_txt}")
    print(f"Total images in final list: {len(final_list)}")

if __name__ == "__main__":
    create_mixed_dataset(
        original_paths_txt="hsface10K_makeup.txt",
        too_few_ids_txt="hsface10K_too_few_examples.txt",
        synthetic_paths_txt="sd_gen_id.txt",
        output_txt="hsface10K_gpt4o_sd.txt"
    )