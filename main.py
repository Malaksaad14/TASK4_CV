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

def process_image(image_path):
    print(f"Processing {image_path}...")
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # 1. Optimal Thresholding
    t_opt = optimal_thresholding(img_array)
    res_opt = apply_threshold(img_array, t_opt)
    
    # 2. Otsu Thresholding
    t_otsu = otsu_thresholding(img_array)
    res_otsu = apply_threshold(img_array, t_otsu)
    
    # 3. Spectral Thresholding (3 modes / 2 thresholds)
    t_spectral = spectral_thresholding(img_array, num_thresholds=2)
    res_spectral = apply_multilevel_threshold(img_array, t_spectral)
    
    # 4. Local Thresholding
    res_local = local_thresholding(img_array, block_size=35, offset=10)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Thresholding Results - {os.path.basename(image_path)}", fontsize=16)
    
    axes[0, 0].imshow(img_array, cmap='gray')
    axes[0, 0].set_title("Original Grayscale")
    
    axes[0, 1].imshow(res_opt, cmap='gray')
    axes[0, 1].set_title(f"Optimal (T={t_opt})")
    
    axes[0, 2].imshow(res_otsu, cmap='gray')
    axes[0, 2].set_title(f"Otsu (T={t_otsu})")
    
    axes[1, 0].imshow(res_spectral, cmap='gray')
    axes[1, 0].set_title(f"Spectral (T1={t_spectral[0]}, T2={t_spectral[1]})")
    
    axes[1, 1].imshow(res_local, cmap='gray')
    axes[1, 1].set_title("Local Thresholding")
    
    axes[1, 2].axis('off') # Placeholder
    
    plt.tight_layout()
    output_name = f"result_{os.path.basename(image_path)}"
    plt.savefig(output_name)
    print(f"Saved results to {output_name}")
    # plt.show()

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
