import numpy as np

def add_dust(img, intensity=0.002):
    """
    Add sparse brownish dust particles to an image.
    Suitable for Mars / outdoor simulation.
    """

    out = img.copy()
    h, w, _ = out.shape
    num_particles = int(h * w * intensity)

    for _ in range(num_particles):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        # Mars-like brown dust (BGR)
        b = np.random.randint(60, 110)
        g = np.random.randint(70, 120)
        r = np.random.randint(120, 180)

        out[y, x] = [b, g, r]

    return out
