import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import random

from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler
from diffusers import StableDiffusionXLPipeline

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

# Resize helper
def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

POSES = [
    "facing forward",                  # frontal (less frequent)
    "facing left",                    # strong side view
    "facing right",                   # strong side view
    "head tilted to the left",         # expressive/angled
    "head tilted to the right",        # expressive/angled
    "chin slightly up",              # subtle vertical
    "glancing sideways",             # off-center gaze
]

POSE_WEIGHTS = [
    0.05,  # facing forward (frontal)
    0.25,  # facing left
    0.25,  # facing right
    0.15,  # tilt left
    0.15,  # tilt right
    0.075, # chin up
    0.075  # glancing sideways
]

EXPRESSIONS = [
    "neutral expression", "slight smile", "serious look", "subtle frown",
    "gentle smile", "relaxed face", "thoughtful gaze", "calm demeanor"
]

LIGHTINGS = [
    "under natural daylight", "lit by a soft lamp", "with backlight from a window",
    "in harsh sunlight", "under fluorescent indoor lighting", "with soft shadows",
    "in golden hour lighting", "with overhead lighting"
]

CAMERA_ANGLES = [
    "tight close-up", "medium close-up", "head and shoulders shot",
    "upper-body shot", "zoomed-out portrait", "cropped at chin",
    "high-angle selfie", "low-angle shot"
]

BACKGROUNDS = [
    "in front of a plain wall", "in a cozy room", "in an office setting",
    "with a blurred background", "in a park", "near a window", "with bookshelf behind",
    "at a cafe", "in a hallway", "with bokeh lights in the background"
]

ACCESSORIES = [
    "", "", "",  # empty strings to keep most prompts accessory-free
    " wearing glasses", " with a scarf", " with a beanie hat",
    " wearing a hoodie", " with earphones", " wearing a turtleneck"
]

def generate_aug_prompt():
    pose = random.choices(POSES, weights=POSE_WEIGHTS, k=1)[0]
    expression = random.choice(EXPRESSIONS)
    lighting = random.choice(LIGHTINGS)
    camera = random.choice(CAMERA_ANGLES)
    background = random.choice(BACKGROUNDS)
    accessory = random.choice(ACCESSORIES)

    prompt = (
        f"A {expression} portrait of the person, {pose}, {camera}, "
        f"{lighting}, {background}{accessory}."
    )
    return prompt

def random_transform_face_kps(image, max_scale=1.2, min_scale=0.8, max_shift=0.1):
    """Randomly resize and shift the keypoint image."""
    w, h = image.size
    scale = random.uniform(min_scale, max_scale)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    image = image.resize((new_w, new_h), resample=Image.BILINEAR)

    # Pad to original size with random shift
    pad_w = max(w - new_w, 0)
    pad_h = max(h - new_h, 0)
    offset_x = random.randint(0, pad_w) if pad_w > 0 else 0
    offset_y = random.randint(0, pad_h) if pad_h > 0 else 0

    new_image = Image.new("RGB", (w, h), (255, 255, 255))
    new_image.paste(image, (offset_x, offset_y))

    return new_image

BASE_IDENTITY_PROMPTS = [
    # Age and Gender
    "a young boy with light skin and freckles, face fully exposed",
    "a teenage girl with tan skin and wavy brown hair, with her face completely visible",
    "an elderly man with a long white beard and glasses, face clearly visible",
    "a middle-aged woman with light skin and straight blonde hair, face fully exposed",
    "a young man with medium brown skin and dreadlocks, with his face uncovered",
    "an elderly woman with dark skin and short curly gray hair, face clearly visible",

    # Ethnicity and Features
    "a South Asian man with a neatly trimmed beard, face exposed and clearly visible",
    "a Hispanic woman with long wavy dark hair, with her face fully exposed",
    "an East Asian person with short straight black hair, face completely visible",
    "a Middle Eastern man with a full beard and a sharp jawline, face fully exposed",
    "a person of Indigenous descent with long braided hair, face clearly visible",
    "a Black woman with medium skin tone and afro-textured hair, face fully exposed",
    "a person of mixed heritage with curly hair and freckles, face clearly visible",

    # Hair Styles and Colors
    "a woman with short platinum blonde hair and bangs, her face fully exposed and visible",
    "a man with long straight black hair tied in a ponytail, face clearly visible",
    "a person with red curly hair and medium skin tone, face fully exposed",
    "a person with blue dyed hair and shaved sides, with their face clearly visible",
    "a man with a bald head and a thick mustache, face fully exposed",
    "a woman with shoulder-length ombre hair and glasses, with her face fully visible",

    # Accessories
    "a person with a round face wearing sunglasses and a hat, with their face clearly visible",
    "a woman with hoop earrings and a nose ring, her face fully exposed",
    "a man with a goatee wearing a baseball cap, face fully exposed",
    "a young person with braces and rectangular glasses, face clearly visible",
    "a person with piercings and colorful hair highlights, face fully exposed",

    # Face Shapes
    "a person with an oval face and light skin, face clearly exposed",
    "a person with a square face and medium brown skin, face fully visible",
    "a person with a heart-shaped face and freckles, face fully exposed",
    "a person with a diamond-shaped face and dark skin, face clearly visible",

    # Expressions
    "a man with a cheerful smile, face fully exposed and visible",
    "a woman with a neutral expression and arched eyebrows, face clearly visible",
    "a young girl laughing with her teeth showing, face fully exposed",
    "a boy with a mischievous grin, face clearly visible",
    "an elderly person with a kind smile and wrinkles, face fully visible",
    "a person with a serious expression and a strong jawline, face exposed",

    # Unique Features
    "a person with vitiligo and short curly hair, face fully exposed",
    "a person with a birthmark on their cheek, face clearly visible",
    "a woman with a beauty mark near her lips, face fully exposed",
    "a man with a scar on his forehead, face clearly visible",
    "a person with heterochromia and medium skin tone, face fully exposed",
    "a person with albinism wearing sunglasses, face clearly visible",

    # Cultural References
    "a South Asian woman wearing a saree and bindi, with her face fully exposed",
    "a Middle Eastern man wearing a keffiyeh and a beard, face fully visible",
    "a person with East Asian features wearing traditional clothing, face clearly visible",
    "a Black man with a shaved head and a dashiki, face fully exposed",
    "a Hispanic woman with a flower crown in her hair, face clearly visible",
    "an Indigenous person with traditional face paint and braids, face fully visible",

    # General
    "a person with long curly hair and a warm smile, face fully exposed",
    "a young man with a round face and freckles, face clearly visible",
    "a woman with light skin and short spiky hair, face fully exposed",
    "a man with medium skin tone and a chiseled jawline, face clearly visible",
    "a teenager with braces and curly dark brown hair, face fully exposed",
    "a person with a medium-length beard and light brown eyes, face clearly visible",
    "a child with dark skin and a big grin, face fully exposed",
]

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Generate 10K unique faces with InstantID")
    parser.add_argument("--id_start", type=int, default=0, help="Start ID number (inclusive)")
    parser.add_argument("--id_end", type=int, default=10000, help="End ID number (exclusive)")
    args = parser.parse_args()
    # Load face detection model
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load ControlNet and InstantID pipeline
    controlnet = ControlNetModel.from_pretrained("./checkpoints/ControlNetModel", torch_dtype=torch.float16)
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        "sd-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe.cuda()

    # Load adapters
    pipe.load_lora_weights("./checkpoints/pytorch_lora_weights.safetensors")
    pipe.fuse_lora()
    pipe.load_ip_adapter_instantid("./checkpoints/ip-adapter.bin")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Load base pipeline
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        "sd-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")
    base_pipe.enable_model_cpu_offload()

    # Output folders
    OUTPUT_DIR = "./sd_xl_ids"
    CROP_DIR = "./sd_xl_crop_ids"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CROP_DIR, exist_ok=True)

    n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"

    target_faces = args.id_end
    identity_idx = args.id_start
    pbar = tqdm(total=target_faces, desc="Generating 10K Unique Faces", initial=identity_idx)

    while identity_idx < target_faces:
        seed = random.randint(0, 999999)
        generator = torch.manual_seed(seed)

        base_prompt = random.choice(BASE_IDENTITY_PROMPTS)
        base_image = base_pipe(
            prompt=base_prompt,
            negative_prompt=n_prompt,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=generator
        ).images[0]

        resized_face = resize_img(base_image)
        face_info = app.get(cv2.cvtColor(np.array(resized_face), cv2.COLOR_RGB2BGR))

        if len(face_info) == 0:
            print(f"No face detected. Skipping.")
            continue

        face_info = sorted(face_info, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        face_emb = face_info['embedding']

        face_kps = draw_kps(resized_face, face_info['kps'])
        face_kps = random_transform_face_kps(face_kps)

        aug_prompt = generate_aug_prompt()
        controlnet_conditioning_scale = random.uniform(0.1, 0.5)
        ip_adapter_scale = random.uniform(0.95, 1.0)

        aug_image = pipe(
            prompt=aug_prompt,
            negative_prompt=n_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=10,
            guidance_scale=1.0,
            generator=generator
        ).images[0]

        id_dir_full = os.path.join(OUTPUT_DIR, f"ID_{identity_idx:05d}")
        id_dir_crop = os.path.join(CROP_DIR, f"ID_{identity_idx:05d}")
        os.makedirs(id_dir_full, exist_ok=True)
        os.makedirs(id_dir_crop, exist_ok=True)

        full_path = os.path.join(id_dir_full, "0.jpg")
        aug_image.save(full_path)

        try:
            detected_faces = app.get(cv2.cvtColor(np.array(aug_image), cv2.COLOR_RGB2BGR))
            if detected_faces:
                best_face = sorted(detected_faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
                x1, y1, x2, y2 = map(int, best_face['bbox'])
                cropped_face = aug_image.crop((x1, y1, x2, y2))
                crop_path = os.path.join(id_dir_crop, "0.jpg")
                cropped_face.save(crop_path)

                # Increment only if face is successfully cropped and saved
                identity_idx += 1
                pbar.update(1)

            else:
                print(f"No face detected in final image. Skipping.")
        except Exception as e:
            print(f"Error cropping face: {e}")