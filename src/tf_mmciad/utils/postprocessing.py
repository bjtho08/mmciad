import numpy as np

def find_contrast_min_and_max(img):
    AUTO_THRESHOLD = 5000
    pixcount = img.size
    limit = pixcount/10
    threshold = pixcount/AUTO_THRESHOLD
    n_bins = 256
    values, _ = np.histogram(img, n_bins)
    i = -1
    found = False
    count = 0
    while True:
        i += 1
        count = values[i]
        if count > limit:
            count = 0
        found = count > threshold
        if found or i >= 255:
            break
    hmin = i
    found = False
    i = 256
    while True:
        i -= 1
        count = values[i]
        if count > limit:
            count = 0
        found = count > threshold
        if found or i < 1:
            break
    hmax = i
    return hmin/256, hmax/256

def auto_contrast(img, order=None):
    from skimage import exposure
    if order is None:
        order = ['b', 'r', 'g']
    output_img = np.zeros_like(np.squeeze(img))
    CHANNEL_INDEX = {'r': 0, 'g': 1, 'b': 2}
    for i, c in enumerate([CHANNEL_INDEX[channel] for channel in order]):
        v_min, v_max = find_contrast_min_and_max(img[..., i])
        if c == 0:
            v_min += 20/256
            v_max = np.percentile(img[..., i], (95.0))
        elif c == 1:
            v_min += 0.1/256
        elif c == 2:
            #v_min *= 1.3
            v_min = max(v_min, np.percentile(img[..., i], (40.0)))
            v_max = np.percentile(img[..., i], (98.0))
        output_img[..., c] = exposure.rescale_intensity(img[..., i], in_range=(v_min, v_max))
    return output_img
    