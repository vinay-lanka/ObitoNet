import os
from PIL import Image
import torch
from tqdm import tqdm
import open_clip
import cv2
from skimage.metrics import structural_similarity as ssim


# Initialize the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-16-plus-240",
    pretrained="laion400m_e32"
)
model.to(device)

print("CLIP model initialized.")

# Function to encode image
def image_encoder(img):
    img = Image.fromarray(img).convert('RGB')
    img = preprocess(img).unsqueeze(0).to(device)
    img = model.encode_image(img)
    return img

# Function to calculate cosine similarity between two images
def calculate_cosine_similarity(img1, img2):
    # Encode the images
    img1_encoded = image_encoder(img1)
    img2_encoded = image_encoder(img2)

    # Normalize the embeddings and calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(img1_encoded, img2_encoded)
    return cos_sim.item()

# Function to calculate SSIM between two images
def calculate_ssim(image1, image2):
    
    # Ensure images are of the same size for SSIM calculation
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    score, _ = ssim(img1, img2, full=True, win_size=7, channel_axis=-1 if image1.ndim == 3 else None)
    return score

# Define paths
set1_path = "/home/apoorv/Documents/CMSC848K/ObitoNet/Dataset/Barn_data/Barn_ply/ply_pov"
set2_path = "/home/apoorv/Documents/CMSC848K/ObitoNet/Dataset/Barn_data/Barn_images"
output_path = "/home/apoorv/Documents/CMSC848K/ObitoNet/Dataset/Barn_data/ply_matched_images"

# Ensure the directories exists
os.makedirs(output_path, exist_ok=True)
os.makedirs(set1_path, exist_ok=True)
os.makedirs(set2_path, exist_ok=True)
print("Directories exist")

# Get the list of images in Set 1 and Set 2
set1_images = sorted([img for img in os.listdir(set1_path) if img.endswith(".png")])
set2_images = sorted([img for img in os.listdir(set2_path) if img.endswith(".jpg")])

# Matching and saving pairs
pair_count = 0

# Similarity threshold
cosine_threshold = 0.5
ssim_threshold = 0.4


for img1_name in tqdm(set1_images, desc="Processing Set 1 Images"):
    img1_path = os.path.join(set1_path, img1_name)
    img1 = cv2.imread(img1_path)

    lowest_similarity = 1.0
    highest_similarity = 0.0

    for img2_name in set2_images:
        img2_path = os.path.join(set2_path, img2_name)
        img2 = cv2.imread(img2_path)

        # Resize the images to 224x224 as required by OpenCLIP
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))

        

        # Calculate cosine similarity
        cosine_similarity = calculate_cosine_similarity(img1, img2)

        # Calculate SSIM
        ssim_similarity = calculate_ssim(img1, img2)

        if cosine_similarity >= cosine_threshold and ssim_similarity >= ssim_threshold:
            # Save the pair
            img1_output = os.path.join(output_path, f"pair{pair_count}.png")
            img2_output = os.path.join(output_path, f"pair{pair_count}.jpg")

            # Make the images larger for better visualization
            img1 = cv2.resize(img1, (540, 540))
            img2 = cv2.resize(img2, (540, 540))

            cv2.imwrite(img1_output, img1)
            cv2.imwrite(img2_output, img2)
            pair_count += 1

            # print(f"Pair {pair_count} saved with similarity: {similarity:.2f}")

            # Update the lowest and highest similarity
            lowest_similarity = min(lowest_similarity, cosine_similarity)
            highest_similarity = max(highest_similarity, cosine_similarity)

    print(f"Lowest similarity: {lowest_similarity:.2f}, Highest similarity: {highest_similarity:.2f}")

print("Number of pairs saved:", pair_count+1)