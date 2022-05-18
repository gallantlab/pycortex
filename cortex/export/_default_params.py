params_flatmap_lateral_medial = {
    "figsize": [16, 9],
    "panels": [
        {
            "extent": [0.000, 0.200, 1.000, 0.800],
            "view": {"angle": "flatmap", "surface": "flatmap"},
        },
        {
            "extent": [0.300, 0.000, 0.200, 0.200],
            "view": {
                "hemisphere": "left",
                "angle": "medial_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.500, 0.000, 0.200, 0.200],
            "view": {
                "hemisphere": "right",
                "angle": "medial_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.000, 0.000, 0.300, 0.300],
            "view": {
                "hemisphere": "left",
                "angle": "lateral_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.700, 0.000, 0.300, 0.300],
            "view": {
                "hemisphere": "right",
                "angle": "lateral_pivot",
                "surface": "inflated",
            },
        },
    ],
}

params_occipital_triple_view = {
    "figsize": [16, 9],
    "panels": [
        {
            "extent": [0.260, 0.000, 0.480, 1.000],
            "view": {
                "angle": "flatmap",
                "surface": "flatmap",
                "zoom": [0.250, 0.000, 0.500, 1.000],
            },
        },
        {
            "extent": [0.000, 0.000, 0.250, 0.333],
            "view": {
                "hemisphere": "left",
                "angle": "bottom_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.000, 0.333, 0.250, 0.333],
            "view": {
                "hemisphere": "left",
                "angle": "medial_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.000, 0.666, 0.250, 0.333],
            "view": {
                "hemisphere": "left",
                "angle": "lateral_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.750, 0.000, 0.250, 0.333],
            "view": {
                "hemisphere": "right",
                "angle": "bottom_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.750, 0.333, 0.250, 0.333],
            "view": {
                "hemisphere": "right",
                "angle": "medial_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.750, 0.666, 0.250, 0.333],
            "view": {
                "hemisphere": "right",
                "angle": "lateral_pivot",
                "surface": "inflated",
            },
        },
    ],
}

params_inflatedless_lateral_medial_ventral = {
    "figsize": [10, 9],
    "panels": [
        {
            "extent": [0.0, 0.0, 0.5, 1 / 3.0],
            "view": {
                "hemisphere": "left",
                "angle": "bottom_pivot",
                "surface": "inflated_less",
            },
        },
        {
            "extent": [0.000, 1 / 3.0, 0.5, 1 / 3.0],
            "view": {
                "hemisphere": "left",
                "angle": "medial_pivot",
                "surface": "inflated_less",
            },
        },
        {
            "extent": [0.000, 2 / 3.0, 0.5, 1 / 3.0],
            "view": {
                "hemisphere": "left",
                "angle": "lateral_pivot",
                "surface": "inflated_less",
            },
        },
        {
            "extent": [0.5, 0.0, 0.5, 1 / 3.0],
            "view": {
                "hemisphere": "right",
                "angle": "bottom_pivot",
                "surface": "inflated_less",
            },
        },
        {
            "extent": [0.5, 1 / 3.0, 0.5, 1 / 3.0],
            "view": {
                "hemisphere": "right",
                "angle": "medial_pivot",
                "surface": "inflated_less",
            },
        },
        {
            "extent": [0.5, 2 / 3.0, 0.5, 1 / 3.0],
            "view": {
                "hemisphere": "right",
                "angle": "lateral_pivot",
                "surface": "inflated_less",
            },
        },
    ],
}


params_inflated_dorsal_lateral_medial_ventral = {
    "figsize": [10, 10],
    "panels": [
        {
            "extent": [0.0, 0.0, 0.5, 0.20],
            "view": {
                "hemisphere": "left",
                "angle": "bottom_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.000, 0.20, 0.5, 0.3],
            "view": {
                "hemisphere": "left",
                "angle": "medial_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.000, 0.50, 0.5, 0.3],
            "view": {
                "hemisphere": "left",
                "angle": "lateral_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.000, 0.80, 0.5, 0.20],
            "view": {
                "hemisphere": "left",
                "angle": "top_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.5, 0.0, 0.5, 0.20],
            "view": {
                "hemisphere": "right",
                "angle": "bottom_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.5, 0.20, 0.5, 0.30],
            "view": {
                "hemisphere": "right",
                "angle": "medial_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.5, 0.50, 0.5, 0.30],
            "view": {
                "hemisphere": "right",
                "angle": "lateral_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.5, 0.80, 0.5, 0.20],
            "view": {
                "hemisphere": "right",
                "angle": "top_pivot",
                "surface": "inflated",
            },
        },
    ],
}

params_flatmap_inflated_lateral_medial_ventral = {
    "figsize": [16, 9],
    "panels": [
        {
            "extent": [0.0, 0.2, 1.0, 0.8],
            "view": {"angle": "flatmap", "surface": "flatmap"},
        },
        {
            "extent": [0.1, 0.05, 0.2, 0.2],
            "view": {
                "hemisphere": "left",
                "angle": "medial_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.7, 0.05, 0.2, 0.2],
            "view": {
                "hemisphere": "right",
                "angle": "medial_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.0, 0.25, 0.2, 0.2],
            "view": {
                "hemisphere": "left",
                "angle": "lateral_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.8, 0.25, 0.2, 0.2],
            "view": {
                "hemisphere": "right",
                "angle": "lateral_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.3, 0.05, 0.15, 0.15],
            "view": {
                "hemisphere": "left",
                "angle": "bottom_pivot",
                "surface": "inflated",
            },
        },
        {
            "extent": [0.55, 0.05, 0.15, 0.15],
            "view": {
                "hemisphere": "right",
                "angle": "bottom_pivot",
                "surface": "inflated",
            },
        },
    ],
}
