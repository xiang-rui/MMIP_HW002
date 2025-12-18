import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def block_stats(x_u16: np.ndarray, blockN: int):
    """Per-block mean/std on a padded image."""
    x = x_u16.astype(np.float32)
    H, W = x.shape
    Hb = H // blockN
    Wb = W // blockN
    mu = np.zeros((Hb, Wb), dtype=np.float32)
    sd = np.zeros((Hb, Wb), dtype=np.float32)
    for br in range(Hb):
        for bc in range(Wb):
            blk = x[br*blockN:(br+1)*blockN, bc*blockN:(bc+1)*blockN]
            mu[br, bc] = blk.mean()
            sd[br, bc] = blk.std()
    return mu, sd

def attenuation_scale(mu_block: np.ndarray, tau=9000.0, kappa=1200.0, alpha=1.5, eps=1e-3):
    """
    Continuous 'importance' from attenuation proxy.
    High attenuation => smaller quant step => smaller scale.
    """
    w = sigmoid((mu_block - tau) / max(1.0, kappa))  # (0,1)
    s_att = 1.0 / (eps + (w ** alpha))
    return s_att.astype(np.float32)

def noise_scale(mu_block: np.ndarray, sd_block: np.ndarray, lam=0.8, c=300.0):
    """
    Poisson-like proxy: relative noise ~ std/(mean+c).
    Noisier => coarser quant => scale increases.
    """
    rel = sd_block / (mu_block + c)
    s_noise = 1.0 + lam * rel
    return s_noise.astype(np.float32)

def stage_freq_matrix(blockN: int, stage_id: int) -> np.ndarray:
    """
    Stage-specific MTF/PSF-inspired frequency weighting m_s(u,v).
    stage_id: 0=DC, 1=low-freq, 2=high/remaining
    """
    params = {
        0: dict(beta=0.10, p=1.0, gamma=0.60),  # protect DC strongly
        1: dict(beta=0.35, p=1.3, gamma=1.00),  # moderate
        2: dict(beta=0.35, p=1.3, gamma=1.05),  # high freq coarser
    }
    beta = params[stage_id]["beta"]
    p    = params[stage_id]["p"]
    gamma= params[stage_id]["gamma"]

    u = np.arange(blockN, dtype=np.float32)
    v = np.arange(blockN, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    denom = np.sqrt(2.0 * (blockN - 1) ** 2)
    rho = np.sqrt(uu*uu + vv*vv) / (denom if denom > 0 else 1.0)  # [0,1]
    m = (1.0 + beta * (rho ** p)) * gamma
    return m.astype(np.float32)

def quantize_block_scale(s_block: np.ndarray, qscale: int = 16):
    """
    Quantize s_block to uint8 for bitstream.
    sb_q = round(s_block * qscale), clipped to [0,255].
    decoder uses s_block = sb_q / qscale.
    """
    sb_q = np.clip(np.round(s_block * qscale), 0, 255).astype(np.uint8)
    return sb_q