import numpy as np

    def generate_ct_phantom(size=512, seed=0, noise_sigma=30.0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)

    yy, xx = np.mgrid[:size, :size]
    cx, cy = size // 2, size // 2

    # soft tissue
    soft = ((xx - cx)**2 / (0.36*size)**2 + (yy - cy)**2 / (0.46*size)**2) <= 1
    img[soft] = 3200.0

    # bone
    bone = ((xx - (cx - 0.12*size))**2 / (0.12*size)**2 + (yy - cy)**2 / (0.18*size)**2) <= 1
    img[bone] = 14000.0

    # organ-like region
    organ = ((xx - (cx + 0.16*size))**2 / (0.18*size)**2 + (yy - (cy + 0.10*size))**2 / (0.12*size)**2) <= 1
    img[organ] = 6500.0

    if noise_sigma > 0:
        img += rng.normal(0.0, noise_sigma, size=(size, size)).astype(np.float32)

    img = np.clip(img, 0, 65535).astype(np.uint16)
    return img

def save_phantom(path="data/phantom_512.npy", size=512, seed=0, noise_sigma=30.0):
    x = generate_ct_phantom(size=size, seed=seed, noise_sigma=noise_sigma)
    np.save(path, x)
    return path

if __name__ == "__main__":
    p = save_phantom()
    print("Saved:", p)