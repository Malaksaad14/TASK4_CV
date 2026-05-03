import numpy as np

def optimal_thresholding(image, delta_t=1.0):
    """
    Optimal Global Thresholding (Iterative)
    """
    # Initial estimate: average intensity
    t = np.mean(image)
    
    while True:
        # Segment into two groups
        g1 = image[image <= t]
        g2 = image[image > t]
        
        if len(g1) == 0 or len(g2) == 0:
            break
            
        # Calculate means
        mu1 = np.mean(g1)
        mu2 = np.mean(g2)
        
        # New threshold
        new_t = (mu1 + mu2) / 2.0
        
        # Check convergence
        if abs(new_t - t) < delta_t:
            t = new_t
            break
        t = new_t
        
    return int(t)

def otsu_thresholding(image):
    """
    Otsu's Thresholding
    Maximizes inter-class variance
    """
    # Histogram
    hist, _ = np.histogram(image, bins=256, range=(0, 255))
    total_pixels = image.size
    
    total_sum = np.sum(np.arange(256) * hist)
    
    sum_b = 0
    w_b = 0
    maximum_variance = 0
    threshold = 0
    
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        
        w_f = total_pixels - w_b
        if w_f == 0:
            break
            
        sum_b += i * hist[i]
        
        m_b = sum_b / w_b
        m_f = (total_sum - sum_b) / w_f
        
        # Between-class variance
        between_variance = w_b * w_f * (m_b - m_f) ** 2
        
        if between_variance > maximum_variance:
            maximum_variance = between_variance
            threshold = i
            
    return threshold

def spectral_thresholding(image, num_thresholds=2):
    """
    Spectral Thresholding (Multilevel Otsu)
    Finds multiple thresholds (default 2 thresholds for 3 modes)
    """
    hist, _ = np.histogram(image, bins=256, range=(0, 255))
    total_pixels = image.size
    
    # Probabilities
    prob = hist / total_pixels
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(np.arange(256) * prob)
    total_mean = cum_mean[-1]
    
    max_variance = -1
    best_thresholds = (0, 0)
    
    # Exhaustive search for 2 thresholds
    # Note: This is O(L^2) which is fine for 256 levels
    for t1 in range(0, 254):
        for t2 in range(t1 + 1, 255):
            # Weights
            w0 = cum_prob[t1]
            w1 = cum_prob[t2] - cum_prob[t1]
            w2 = 1.0 - cum_prob[t2]
            
            if w0 == 0 or w1 == 0 or w2 == 0:
                continue
            
            # Means
            m0 = cum_mean[t1] / w0
            m1 = (cum_mean[t2] - cum_mean[t1]) / w1
            m2 = (total_mean - cum_mean[t2]) / w2
            
            # Variance
            variance = w0 * (m0 - total_mean)**2 + w1 * (m1 - total_mean)**2 + w2 * (m2 - total_mean)**2
            
            if variance > max_variance:
                max_variance = variance
                best_thresholds = (t1, t2)
                
    return best_thresholds

def local_thresholding(image, block_size=15, offset=10):
    """
    Local (Adaptive) Thresholding
    Uses an integral image for efficiency
    """
    rows, cols = image.shape
    image_float = image.astype(float)
    
    # Compute integral image
    integral = np.zeros((rows + 1, cols + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(image_float, axis=0), axis=1)
    
    result = np.zeros_like(image, dtype=np.uint8)
    
    pad = block_size // 2
    
    for r in range(rows):
        for c in range(cols):
            # Define window boundaries
            r1, r2 = max(0, r - pad), min(rows - 1, r + pad)
            c1, c2 = max(0, c - pad), min(cols - 1, c + pad)
            
            # Use integral image to get sum in O(1)
            # Area: [r1, c1] to [r2, c2]
            count = (r2 - r1 + 1) * (c2 - c1 + 1)
            s = integral[r2 + 1, c2 + 1] - integral[r1, c2 + 1] - integral[r2 + 1, c1] + integral[r1, c1]
            
            local_mean = s / count
            
            if image[r, c] > local_mean - offset:
                result[r, c] = 255
            else:
                result[r, c] = 0
                
    return result

def apply_threshold(image, t):
    """Helper to apply a single threshold"""
    return (image > t).astype(np.uint8) * 255

def apply_multilevel_threshold(image, thresholds):
    """Helper to apply multiple thresholds (labels 0, 127, 255 for 3 modes)"""
    t1, t2 = sorted(thresholds)
    result = np.zeros_like(image, dtype=np.uint8)
    result[image > t1] = 127
    result[image > t2] = 255
    return result
