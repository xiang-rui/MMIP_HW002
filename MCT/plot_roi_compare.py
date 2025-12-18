import numpy as np
import matplotlib.pyplot as plt
from roi import roi_mask_from_phantom

# Load images
x = np.load("data/phantom_512.npy")
y_on  = np.load("results/q10_v3_s3.npy")
y_off = np.load("results/q10_v3_no_roi_s3.npy")

roi = roi_mask_from_phantom(x, bone_threshold=9000)

imgs = [
    y_on,
    y_off,
    roi.astype(np.uint16) * 65535
]

titles = [
    "ROI-aware (ON)",
    "Uniform Quantization (ROI-OFF)",
    "ROI Mask"
]

plt.figure(figsize=(8, 3))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(imgs[i], cmap="gray", vmin=0, vmax=65535)
    plt.title(titles[i], fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/fig_roi_compare.png", dpi=300)
plt.show()
