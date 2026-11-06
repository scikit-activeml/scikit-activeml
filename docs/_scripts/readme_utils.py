"""Shared helpers for generating the README animations."""

import numpy as np

from matplotlib import animation
from PIL import Image


class SharedPaletteWriter(animation.PillowWriter):
    """Map all frames to one palette that keeps `colors` exact."""

    def __init__(self, colors, **kwargs):
        super().__init__(**kwargs)
        self.colors = colors

    def finish(self):
        frames = np.stack([np.asarray(f.convert("RGB")) for f in self._frames])
        codes = frames.astype(np.int32) @ np.array([65536, 256, 1])
        values, inverse, counts = np.unique(
            codes, return_inverse=True, return_counts=True
        )
        exact = [int(color[1:], 16) for color in self.colors]
        frequent = values[np.argsort(-counts)].tolist()
        # One index stays free, so that the GIF can store unchanged
        # pixels as transparent.
        palette = list(dict.fromkeys(exact + frequent))[:255]
        palette = np.array(
            [[p >> 16, (p >> 8) & 255, p & 255] for p in palette]
        )
        values = np.stack([values >> 16, (values >> 8) & 255, values & 255], 1)
        distances = ((values[:, None] - palette[None]) ** 2).sum(-1)
        indices = distances.argmin(1)[inverse].reshape(codes.shape)
        self._frames = []
        for frame in indices.astype(np.uint8):
            image = Image.fromarray(frame, "P")
            image.putpalette(palette.ravel().tolist())
            self._frames.append(image)
        super().finish()
