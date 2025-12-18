import numpy as np
import matplotlib.pyplot as plt

# Load images
x  = np.load("data/phantom_512.npy")
s1 = np.load("results/q10_v3_s1.npy")
s2 = np.load("results/q10_v3_s2.npy")
s3 = np.load("results/q10_v3_s3.npy")

imgs = [x, s1, s2, s3]
titles = [
    "Original",
    "Stage 1 (DC only)",
    "Stage 2 (Low-frequency)",
    "Stage 3 (Full reconstruction)"
]

plt.figure(figsize=(10, 3))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(imgs[i], cmap="gray", vmin=0, vmax=65535)
    plt.title(titles[i], fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/fig_progressive.png", dpi=300)
plt.show()
