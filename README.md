# README
This is the code of Ferry_Li for the 4th DataCV Challenge in conjunction with ICCV workshop 2025: [Hybrid Generative Fusion for Efficient and Privacy-Preserving Face Recognition Dataset Generation](https://arxiv.org/abs/2508.10672)

## FRAMEWORK


## Dataset
The 10K-scale training set is [here](https://drive.google.com/file/d/1Lw09rwuVQN8jOYjx2YJQN431TuCECfZn/view?usp=drive_link).

## Steps
1. Step 1: Use `feature_extractor.py` provided in [https://github.com/HaiyuWu/Vec2Face](https://github.com/HaiyuWu/Vec2Face) to extract image embeddings from the current [HSFace dataset](https://huggingface.co/datasets/BooBooWu/Vec2Face/tree/main/HSFaces).

2. Step 2: Run `reduce_hsface.py` to record only consistent identity image paths in the HSFace dataset.

3. Step 3: Run `gpt_clean_parallel.py` to record inconsistent identity image paths in the HSFace dataset.

4. Step 4: Run `convert.py` to convert the `json` format to `txt` format.

5. Step 5: Run `hsface_makeup.py` to augment images to 50 per identity, and save the augmented images.

6. Step 6: Run `generate_id.py` to generate a new image for one identity based on various prompts.

7. Step 7: Use [image_generation_with_reference.py]([https://github.com/HaiyuWu/Vec2Face](https://github.com/HaiyuWu/Vec2Face/blob/main/image_generation_with_reference.py)) to expand one image to 50 per identity. 

8. Step 8: Run `merge_dataset.py` to merge the cleaned HSFace dataset and diffusion-Vec2Face-expanded dataset. 
