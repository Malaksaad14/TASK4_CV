import numpy as np
import cv2
from sklearn.cluster import KMeans, AgglomerativeClustering

def kmeans_segmentation(image, n_clusters=3):
    """
    Unsupervised segmentation using K-means clustering.
    Works for both Grayscale and RGB images.
    """
    if len(image.shape) == 3:
        pixel_values = image.reshape((-1, 3))
    else:
        pixel_values = image.reshape((-1, 1))
        
    pixel_values = np.float32(pixel_values)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_values)
    centers = np.uint8(kmeans.cluster_centers_)
    
    # Reconstruct the segmented image
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

def mean_shift_segmentation(image, spatial_radius=20, color_radius=40):
    """
    Unsupervised segmentation using Mean Shift.
    We use cv2.pyrMeanShiftFiltering for efficiency on images.
    """
    if len(image.shape) == 3:
        # Convert to BGR if it's RGB for OpenCV (assuming input is RGB or Gray)
        # Assuming input is already in the format we want to process, 
        # pyrMeanShiftFiltering expects an 8-bit, 3-channel image.
        # If it's already 3-channel, just apply it.
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        shifted = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius)
        return shifted
    else:
        # For grayscale, convert to BGR, apply meanshift, convert back
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        shifted = cv2.pyrMeanShiftFiltering(bgr, spatial_radius, color_radius)
        return cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

def region_growing_segmentation(image, seed_point=None, threshold=20):
    """
    Region growing segmentation.
    Uses cv2.floodFill starting from a seed point.
    """
    img = image.copy()
        
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    if seed_point is None:
        # Start from the center if no seed point is provided
        seed_point = (w // 2, h // 2)
        
    # Ensure seed_point is within bounds
    seed_point = (min(max(seed_point[0], 0), w - 1), min(max(seed_point[1], 0), h - 1))
    
    if len(image.shape) == 3:
        # Color image flood fill
        diff = (threshold, threshold, threshold)
        cv2.floodFill(img, mask, seed_point, (0, 255, 0), diff, diff, 4 | (255 << 8))
    else:
        # Grayscale flood fill
        # We can fill with a specific color by converting to BGR or just fill with a gray value
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        diff = (threshold, threshold, threshold)
        cv2.floodFill(img_bgr, mask, seed_point, (0, 255, 0), diff, diff, 4 | (255 << 8))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
    return img

def agglomerative_segmentation(image, n_clusters=3):
    """
    Unsupervised segmentation using Agglomerative Clustering.
    Downsamples the image first because Agglomerative is O(N^2) or O(N^3).
    """
    # Downsample to speed up (50x50 is manageable)
    scale_factor = 50.0 / max(image.shape[:2])
    if scale_factor < 1.0:
        small_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        small_img = cv2.resize(image, small_size)
    else:
        small_img = image.copy()
        
    if len(small_img.shape) == 3:
        pixel_values = small_img.reshape((-1, 3))
    else:
        pixel_values = small_img.reshape((-1, 1))
        
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(pixel_values)
    
    # Calculate centers manually
    labels_2d = labels.reshape(small_img.shape[:2])
    
    # Map labels back to original size
    large_labels = cv2.resize(labels_2d.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create segmented image based on cluster means
    segmented = np.zeros_like(image)
    for i in range(n_clusters):
        mask = large_labels == i
        if len(image.shape) == 3:
            mean_color = cv2.mean(image, mask=mask.astype(np.uint8))[:3]
            segmented[mask] = mean_color
        else:
            mean_color = cv2.mean(image, mask=mask.astype(np.uint8))[0]
            segmented[mask] = mean_color
            
    return segmented
