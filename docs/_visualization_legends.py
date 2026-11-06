"""Legend definitions and HTML rendering for documentation visualizations."""

import base64

from matplotlib import colormaps
from matplotlib.colors import to_hex


def _swatch(shape):
    """Draw a legend shape on a white chip, like the animations' background.

    The chip is an image because notebook sanitizers remove inline SVG.
    """
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="20" '
        'viewBox="0 0 32 20"><rect x="0.5" y="0.5" width="31" height="19" '
        'rx="3" fill="#ffffff" stroke="#d1d5da"/>' + shape + "</svg>"
    )
    data = base64.b64encode(svg.encode()).decode()
    return (
        f'<img src="data:image/svg+xml;base64,{data}" width="32" height="20" '
        'alt="">'
    )


def _line(color, width=2, opacity=1, dashed=False):
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return (
        f'<path d="M5 10 H27" stroke="{color}" stroke-width="{width}" '
        f'stroke-opacity="{opacity}"{dash}/>'
    )


def _dot(color, radius=3, edge="none"):
    return (
        f'<circle cx="16" cy="10" r="{radius}" fill="{color}" '
        f'stroke="{edge}"/>'
    )


def _halo(color):
    return (
        f'<circle cx="16" cy="10" r="7" fill="{color}" fill-opacity="0.28" '
        f'stroke="{color}" stroke-width="2"/>'
    )


def _ring(color):
    return (
        f'<circle cx="16" cy="10" r="5" fill="none" stroke="{color}" '
        f'stroke-width="1.2"/>'
    )


def _x_marker(color):
    return (
        '<path d="M12 7.6 L13.6 6 L16 8.4 L18.4 6 L20 7.6 L17.6 10 L20 12.4 '
        'L18.4 14 L16 11.6 L13.6 14 L12 12.4 L14.4 10 Z" '
        f'fill="{color}" stroke="{_LINE_COLOR}" stroke-width="0.8"/>'
    )


def _bar(color):
    return f'<rect x="7" y="6" width="18" height="8" fill="{color}"/>'


def _tick(color):
    return f'<path d="M16 4 V16" stroke="{color}" stroke-width="2"/>'


def _ramp(cmap, opacity):
    colors = colormaps[cmap]([0.0, 0.25, 0.5, 0.75, 1.0])
    return "".join(
        f'<rect x="{4 + 4.8 * i:g}" y="5" width="4.8" height="10" '
        f'fill="{to_hex(color)}" fill-opacity="{opacity}"/>'
        for i, color in enumerate(colors)
    )


def _stems(color):
    return "".join(
        f'<path d="M{x} 16 V{top}" stroke="{color}" stroke-width="2"/>'
        for x, top in [(6, 11), (10, 7), (14, 5), (18, 9), (22, 6), (26, 12)]
    )


def _band(color, opacity):
    return (
        f'<rect x="4" y="5" width="24" height="10" fill="{color}" '
        f'fill-opacity="{opacity}"/>'
    )


_CLASS_COLORS = ["#2a78d6", "#1baf7a"]
_LINE_COLOR = "#222832"
_SELECTION_COLOR = "#eb6834"


def _samples(values):
    """Legend entries of the sample markers for the two label values."""
    return [
        *[
            (_swatch(_dot("none", edge=color)), f"Unlabeled sample {value}")
            for value, color in zip(values, _CLASS_COLORS)
        ],
        *[
            (
                _swatch(_dot(color, radius=4.5, edge=_LINE_COLOR)),
                f"Labeled sample {value}",
            )
            for value, color in zip(values, _CLASS_COLORS)
        ],
    ]


def _boundaries(confidences, utility="Utility"):
    """Legend entries of the decision boundary, the confidence lines for the
    two label values, and the utilities."""
    return [
        (_swatch(_line(_LINE_COLOR)), "Decision boundary"),
        *[
            (_swatch(_line(color, dashed=True)), f"75% confidence {value}")
            for value, color in zip(confidences, _CLASS_COLORS)
        ],
        (_swatch(_ramp("Purples", 0.35)), f"{utility}, from low to high"),
    ]


_CLASSES = ["of class 0", "of class 1"]
_CLASS_CONFIDENCES = ["for class 0", "for class 1"]
_SELECTION = (
    _swatch(_halo(_SELECTION_COLOR)),
    "Sample selected in this cycle",
)

# The legend of each visualization group. Keep the swatches in line with the
# plots of the group's templates.
VISUALIZATION_LEGENDS = {
    "pool-classification": [
        *_samples(_CLASSES),
        _SELECTION,
        *_boundaries(_CLASS_CONFIDENCES),
    ],
    "pool-classification-batch": [
        *_samples(_CLASSES),
        (_swatch(_halo(_SELECTION_COLOR)), "Selection of the panel"),
        (
            _swatch(_ring(_SELECTION_COLOR)),
            "Earlier selection of the batch",
        ),
        *_boundaries(_CLASS_CONFIDENCES),
    ],
    "pool-multilabel": [
        *_samples(["without the label", "with the label"]),
        _SELECTION,
        *_boundaries(
            ["that the label is absent", "that the label is present"]
        ),
    ],
    "pool-regression": [
        (_swatch(_dot("none", edge="#8c8c8c")), "Unlabeled sample"),
        (
            _swatch(_dot(_CLASS_COLORS[0], radius=4.5, edge=_LINE_COLOR)),
            "Labeled sample",
        ),
        _SELECTION,
        (_swatch(_line("#8c8c8c", width=1.5, dashed=True)), "True function"),
        (_swatch(_line(_LINE_COLOR)), "Prediction"),
        (
            _swatch(_band(_LINE_COLOR, 0.12)),
            "Prediction ± one standard deviation, if available",
        ),
        (_swatch(_stems("#8c6bb1")), "Utility, from low to high"),
        (
            _swatch(_tick(_SELECTION_COLOR)),
            "Positions of the selected samples",
        ),
    ],
    "pool-multi-annotator": [
        *_samples(_CLASSES)[:2],
        *[
            (
                _swatch(_dot(color, radius=4.5, edge=_LINE_COLOR)),
                f"Correct majority vote for class {label}",
            )
            for label, color in enumerate(_CLASS_COLORS)
        ],
        *[
            (
                _swatch(_x_marker(color)),
                f"Wrong majority vote for class {label}",
            )
            for label, color in enumerate(_CLASS_COLORS)
        ],
        _SELECTION,
        *_boundaries(
            _CLASS_CONFIDENCES, utility="Mean utility over annotators"
        ),
        (_swatch(_bar("#b9bec6")), "Labels requested earlier"),
        (
            _swatch(_bar(_SELECTION_COLOR)),
            "Labels requested in this cycle",
        ),
        (_swatch(_tick(_LINE_COLOR)), "True annotator accuracy"),
        (
            _swatch(_dot(_LINE_COLOR, radius=3.5)),
            "Estimated annotator accuracy",
        ),
    ],
    "stream-classification": [
        *_samples(_CLASSES),
        (
            _swatch(_halo(_SELECTION_COLOR)),
            "Label queried since the previous frame",
        ),
        (_swatch(_line(_LINE_COLOR)), "Decision boundary over time"),
        (_swatch(_stems("#8c6bb1")), "Utility, label not queried"),
        (_swatch(_stems(_SELECTION_COLOR)), "Utility, label queried"),
        (
            _swatch(_line("#8c8c8c", width=1.5, dashed=True)),
            "Budget of the strategy",
        ),
    ],
}


# The visualization groups whose animations show a learning curve. Their
# legends add a note on its noise.
LEARNING_CURVE_GROUPS = (
    "pool-classification",
    "pool-classification-batch",
    "pool-multilabel",
    "pool-multi-annotator",
    "pool-regression",
    "stream-classification",
)


def visualization_legend_html(group):
    """Render the legend of a visualization group as HTML without blank
    lines, so that notebooks show it as one HTML block."""
    items = "\n".join(
        f"<li>{swatch}{label}</li>"
        for swatch, label in VISUALIZATION_LEGENDS[group]
    )
    note = (
        '<p class="visualization-legend-note">The learning curve shows a '
        "single run on a small synthetic dataset. Runs with other data or "
        "seeds can give noticeably different curves.</p>\n"
        if group in LEARNING_CURVE_GROUPS
        else ""
    )
    return (
        '<div class="visualization-legend">\n'
        '<p class="visualization-legend-title kicker">Legend</p>\n'
        f'<ul class="visualization-legend-items">\n{items}\n</ul>\n'
        f"{note}</div>"
    )
