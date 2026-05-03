import numpy as np

def kmeans_segmentation(image, n_clusters=3, max_iters=100):
    """
    K-Means Segmentation from scratch using NumPy.
    """
    is_color = len(image.shape) == 3
    if is_color:
        pixels = image.reshape((-1, 3)).astype(np.float32)
    else:
        pixels = image.reshape((-1, 1)).astype(np.float32)
        
    np.random.seed(42)
    random_indices = np.random.choice(pixels.shape[0], n_clusters, replace=False)
    centroids = pixels[random_indices]
    
    labels = np.zeros(pixels.shape[0], dtype=np.int32)
    
    for _ in range(max_iters):
        # Distances: shape (N, K)
        distances = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        
        if np.all(labels == new_labels):
            break
        labels = new_labels
        
        for k in range(n_clusters):
            mask = (labels == k)
            if np.any(mask):
                centroids[k] = np.mean(pixels[mask], axis=0)
                
    segmented = centroids[labels].reshape(image.shape).astype(np.uint8)
    return segmented

def region_growing_segmentation(image, seed_point=None, threshold=20):
    """
    Region Growing Segmentation from scratch using a Queue (Breadth-First Search).
    """
    h, w = image.shape[:2]
    is_color = len(image.shape) == 3
    
    if seed_point is None:
        seed_point = (w // 2, h // 2)
        
    sx, sy = seed_point
    sx = max(0, min(sx, w - 1))
    sy = max(0, min(sy, h - 1))
    
    visited = np.zeros((h, w), dtype=bool)
    
    # Queue for BFS
    queue = [(sx, sy)]
    visited[sy, sx] = True
    
    seed_val = image[sy, sx].astype(np.float32)
    fill_color = np.array([0, 255, 0], dtype=np.uint8) if is_color else 255
    
    region_pixels = []
    
    # Python loops can be slow, but it's required for 'from scratch' Region Growing
    while queue:
        cx, cy = queue.pop(0)
        region_pixels.append((cy, cx))
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                val = image[ny, nx].astype(np.float32)
                
                if is_color:
                    dist = np.linalg.norm(val - seed_val)
                else:
                    dist = abs(val - seed_val)
                    
                if dist <= threshold:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
                    
    result = image.copy()
    for cy, cx in region_pixels:
        result[cy, cx] = fill_color
        
    return result

def agglomerative_segmentation(image, n_clusters=3):
    """
    Agglomerative Hierarchical Clustering from scratch.
    Downsamples the image heavily to make O(N^3) distance math feasible.
    """
    h, w = image.shape[:2]
    
    # Downsample to a small grid (e.g. max 30x30) to compute from scratch
    scale_y = max(1, h // 30)
    scale_x = max(1, w // 30)
    
    small = image[::scale_y, ::scale_x]
    sh, sw = small.shape[:2]
    N = sh * sw
    
    is_color = len(image.shape) == 3
    if is_color:
        data = small.reshape((N, 3)).astype(np.float32)
    else:
        data = small.reshape((N, 1)).astype(np.float32)
        
    clusters = {i: [i] for i in range(N)}
    
    dist_matrix = np.linalg.norm(data[:, None] - data[None, :], axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    
    active_clusters = list(range(N))
    
    # Single Linkage clustering
    while len(active_clusters) > n_clusters:
        sub_dist = dist_matrix[np.ix_(active_clusters, active_clusters)]
        min_idx = np.argmin(sub_dist)
        r, c = np.unravel_index(min_idx, sub_dist.shape)
        
        if r == c: # Fallback if error
            break
            
        c1, c2 = active_clusters[r], active_clusters[c]
        
        clusters[c1].extend(clusters[c2])
        del clusters[c2]
        active_clusters.remove(c2)
        
        for other in active_clusters:
            if other != c1:
                new_dist = min(dist_matrix[c1, other], dist_matrix[c2, other])
                dist_matrix[c1, other] = new_dist
                dist_matrix[other, c1] = new_dist
                
    # Upsample map to original size
    small_labels = np.zeros((sh, sw), dtype=np.int32)
    for label_idx, c_id in enumerate(active_clusters):
        for pt_idx in clusters[c_id]:
            sy, sx = pt_idx // sw, pt_idx % sw
            small_labels[sy, sx] = label_idx
            
    large_labels = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            sy = min(y // scale_y, sh - 1)
            sx = min(x // scale_x, sw - 1)
            large_labels[y, x] = small_labels[sy, sx]
            
    segmented = np.zeros_like(image)
    for i in range(len(active_clusters)):
        mask = (large_labels == i)
        if np.any(mask):
            mean_color = np.mean(image[mask], axis=0)
            segmented[mask] = mean_color
            
    return segmented

def mean_shift_segmentation(image, spatial_radius=20, color_radius=40, max_iters=15):
    """
    Mean Shift Segmentation from scratch.
    Sub-samples pixels to find modes, then maps all pixels to the nearest mode.
    """
    h, w = image.shape[:2]
    is_color = len(image.shape) == 3
    
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    if is_color:
        features = np.zeros((h, w, 5), dtype=np.float32)
        features[:, :, :3] = image
        features[:, :, 3] = y_coords
        features[:, :, 4] = x_coords
    else:
        features = np.zeros((h, w, 3), dtype=np.float32)
        features[:, :, 0] = image
        features[:, :, 1] = y_coords
        features[:, :, 2] = x_coords
        
    flat_features = features.reshape((-1, features.shape[2]))
    
    # Pick random points as initial mode seeds (e.g. 200 points) to speed up
    np.random.seed(42)
    num_samples = min(200, flat_features.shape[0])
    sample_indices = np.random.choice(flat_features.shape[0], num_samples, replace=False)
    modes = flat_features[sample_indices].copy()
    
    # Iterate to find converged modes
    for i in range(num_samples):
        mode = modes[i]
        for _ in range(max_iters):
            if is_color:
                color_dist = np.linalg.norm(flat_features[:, :3] - mode[:3], axis=1)
                spatial_dist = np.linalg.norm(flat_features[:, 3:] - mode[3:], axis=1)
            else:
                color_dist = np.abs(flat_features[:, 0] - mode[0])
                spatial_dist = np.linalg.norm(flat_features[:, 1:] - mode[1:], axis=1)
                
            in_window = (color_dist < color_radius) & (spatial_dist < spatial_radius)
            if not np.any(in_window):
                break
                
            new_mode = np.mean(flat_features[in_window], axis=0)
            if np.linalg.norm(new_mode - mode) < 0.5:
                break
            mode = new_mode
        modes[i] = mode
        
    # Merge similar modes
    unique_modes = []
    for mode in modes:
        is_unique = True
        for umode in unique_modes:
            if is_color:
                c_dist = np.linalg.norm(mode[:3] - umode[:3])
                s_dist = np.linalg.norm(mode[3:] - umode[3:])
            else:
                c_dist = abs(mode[0] - umode[0])
                s_dist = np.linalg.norm(mode[1:] - umode[1:])
            
            if c_dist < color_radius / 2 and s_dist < spatial_radius / 2:
                is_unique = False
                break
        if is_unique:
            unique_modes.append(mode)
            
    unique_modes = np.array(unique_modes)
    
    # Reconstruct Image
    segmented = np.zeros_like(image)
    chunk_size = 20000
    
    for start in range(0, flat_features.shape[0], chunk_size):
        end = min(start + chunk_size, flat_features.shape[0])
        chunk = flat_features[start:end]
        
        if is_color:
            c_diff = chunk[:, None, :3] - unique_modes[None, :, :3]
            s_diff = chunk[:, None, 3:] - unique_modes[None, :, 3:]
            dist = np.linalg.norm(c_diff / color_radius, axis=2) + np.linalg.norm(s_diff / spatial_radius, axis=2)
        else:
            c_diff = chunk[:, None, 0:1] - unique_modes[None, :, 0:1]
            s_diff = chunk[:, None, 1:] - unique_modes[None, :, 1:]
            dist = np.abs(c_diff[:, :, 0] / color_radius) + np.linalg.norm(s_diff / spatial_radius, axis=2)
            
        labels = np.argmin(dist, axis=1)
        
        if is_color:
            segmented.reshape(-1, 3)[start:end] = unique_modes[labels, :3]
        else:
            segmented.reshape(-1)[start:end] = unique_modes[labels, 0]
            
    return segmented
