import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from thresholding import (
    optimal_thresholding,
    otsu_thresholding,
    spectral_thresholding,
    local_thresholding,
    apply_threshold,
    apply_multilevel_threshold
)

from segmentation import (
    kmeans_segmentation,
    mean_shift_segmentation,
    region_growing_segmentation,
    agglomerative_segmentation
)

def process_image(image_path):
    print(f"Processing {image_path}...")
    
    # --- PART A: Thresholding ---
    img_gray_pil = Image.open(image_path).convert('L')
    img_gray = np.array(img_gray_pil)
    
    # 1. Optimal Thresholding
    t_opt = optimal_thresholding(img_gray)
    res_opt = apply_threshold(img_gray, t_opt)
    
    # 2. Otsu Thresholding
    t_otsu = otsu_thresholding(img_gray)
    res_otsu = apply_threshold(img_gray, t_otsu)
    
    # 3. Spectral Thresholding
    t_spectral = spectral_thresholding(img_gray, num_thresholds=2)
    res_spectral = apply_multilevel_threshold(img_gray, t_spectral)
    
    # 4. Local Thresholding
    res_local = local_thresholding(img_gray, block_size=35, offset=10)
    
    # Visualization Part A
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Thresholding Results - {os.path.basename(image_path)}", fontsize=16)
    
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title("Original Grayscale")
    
    axes[0, 1].imshow(res_opt, cmap='gray')
    axes[0, 1].set_title(f"Optimal (T={t_opt})")
    
    axes[0, 2].imshow(res_otsu, cmap='gray')
    axes[0, 2].set_title(f"Otsu (T={t_otsu})")
    
    axes[1, 0].imshow(res_spectral, cmap='gray')
    axes[1, 0].set_title(f"Spectral (T1={t_spectral[0]}, T2={t_spectral[1]})")
    
    axes[1, 1].imshow(res_local, cmap='gray')
    axes[1, 1].set_title("Local Thresholding")
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_name_a = f"result_thresholding_{os.path.basename(image_path)}"
    plt.savefig(output_name_a)
    print(f"Saved thresholding results to {output_name_a}")
    plt.close()

    # --- PART B: Segmentation ---
    img_color_pil = Image.open(image_path).convert('RGB')
    img_color = np.array(img_color_pil)
    
    print(f"  -> Running K-Means...")
    res_kmeans = kmeans_segmentation(img_color, n_clusters=4)
    
    print(f"  -> Running Mean Shift...")
    res_meanshift = mean_shift_segmentation(img_color, spatial_radius=20, color_radius=40)
    
    print(f"  -> Running Region Growing...")
    res_region = region_growing_segmentation(img_color, threshold=30)
    
    print(f"  -> Running Agglomerative...")
    res_agglo = agglomerative_segmentation(img_color, n_clusters=4)
    
    # Visualization Part B
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Segmentation Results - {os.path.basename(image_path)}", fontsize=16)
    
    axes[0, 0].imshow(img_color)
    axes[0, 0].set_title("Original Color")
    
    axes[0, 1].imshow(res_kmeans)
    axes[0, 1].set_title("K-Means (K=4)")
    
    axes[0, 2].imshow(res_meanshift)
    axes[0, 2].set_title("Mean Shift")
    
    axes[1, 0].imshow(res_region)
    axes[1, 0].set_title("Region Growing")
    
    axes[1, 1].imshow(res_agglo)
    axes[1, 1].set_title("Agglomerative (K=4)")
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_name_b = f"result_segmentation_{os.path.basename(image_path)}"
    plt.savefig(output_name_b)
    print(f"Saved segmentation results to {output_name_b}")
    plt.close()

if __name__ == "__main__":
    image_dir = "images"
    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} folder not found.")
    else:
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print("No images found in images folder.")
        for img_name in images:
            process_image(os.path.join(image_dir, img_name))
