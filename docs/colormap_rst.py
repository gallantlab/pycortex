"""
This will make the colormaps.rst file for the docs page.
"""
import os

path_to_colormaps = "../filestore/colormaps/"


def is_2d(name):
    """Return True if the colormap image is 2-dimensional (square)."""
    import matplotlib.pyplot as plt
    path = os.path.join(path_to_colormaps, name + ".png")
    try:
        img = plt.imread(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read colormap image '{path}': {exc}"
        ) from exc
    return img.shape[0] > 1


# ---------------------------------------------------------------------------
# Category definitions
# Each entry is (display_name, description, [list_of_colormap_names])
# Reversed variants (_r suffix) are grouped with their base colormap where
# possible, but are also listed individually so users can find them.
# ---------------------------------------------------------------------------

CATEGORIES_1D = [
    (
        "Perceptually Uniform Sequential",
        "These colormaps change in brightness and/or hue uniformly, making "
        "them well-suited for representing ordered data.  They are "
        "distinguishable by people with colour-vision deficiency and print "
        "well in greyscale.  These are the **recommended defaults** for most "
        "sequential data.",
        [
            "viridis", "viridis_r",
            "plasma", "plasma_r",
            "inferno", "inferno_r",
            "magma", "magma_r",
        ],
    ),
    (
        "Sequential",
        "Sequential colormaps transition from light to dark (or vice versa) "
        "through a single hue or a small range of hues.  They are suitable "
        "for data that progresses from low to high values.",
        [
            "Blues", "Blues_r",
            "BuGn", "BuGn_r",
            "BuPu", "BuPu_r",
            "GnBu", "GnBu_r",
            "Greens", "Greens_r",
            "Greys", "Greys_r",
            "Oranges", "Oranges_r",
            "OrRd", "OrRd_r",
            "PuBu", "PuBu_r",
            "PuBuGn", "PuBuGn_r",
            "PuRd", "PuRd_r",
            "Purples", "Purples_r",
            "RdPu", "RdPu_r",
            "Reds", "Reds_r",
            "YlGn", "YlGn_r",
            "YlGnBu", "YlGnBu_r",
            "YlOrBr", "YlOrBr_r",
            "YlOrRd", "YlOrRd_r",
            "afmhot", "afmhot_r",
            "autumn", "autumn_r",
            "autumn_blkmin",
            "autumnblack",
            "binary", "binary_r",
            "bone", "bone_r",
            "cool", "cool_r",
            "copper", "copper_r",
            "cubehelix", "cubehelix_r",
            "fire",
            "gist_earth", "gist_earth_r",
            "gist_gray", "gist_gray_r",
            "gist_heat", "gist_heat_r",
            "gist_stern", "gist_stern_r",
            "gist_yarg", "gist_yarg_r",
            "gnuplot", "gnuplot_r",
            "gnuplot2", "gnuplot2_r",
            "gray", "gray_r",
            "hot", "hot_r",
            "ocean", "ocean_r",
            "pink", "pink_r",
            "spring", "spring_r",
            "summer", "summer_r",
            "terrain", "terrain_r",
            "winter", "winter_r",
        ],
    ),
    (
        "Diverging",
        "Diverging colormaps have a bright neutral midpoint and darken "
        "towards two contrasting hues at either end.  They are ideal when "
        "data has a meaningful centre value (e.g. zero for "
        "positive/negative contrasts).",
        [
            "BrBG", "BrBG_r",
            "PRGn", "PRGn_r",
            "PiYG", "PiYG_r",
            "PuOr", "PuOr_r",
            "RdBu", "RdBu_r",
            "RdGy", "RdGy_r",
            "RdYlBu", "RdYlBu_r",
            "RdYlGn", "RdYlGn_r",
            "Spectral", "Spectral_r",
            "bwr", "bwr_r",
            "coolwarm", "coolwarm_r",
            "seismic", "seismic_r",
            "BuBkRd",
            "BuWtRd",
            "GreenWhiteBlue",
            "GreenWhiteRed",
        ],
    ),
    (
        "Cyclic",
        "Cyclic colormaps have identical colours at both endpoints and are "
        "appropriate for data that wraps around, such as phase, orientation, "
        "or polar angle (e.g. retinotopic maps).",
        [
            "hsv", "hsv_r",
            "Retinotopy_RYBCR",
            "RGrB_tsi",
        ],
    ),
    (
        "Qualitative",
        "Qualitative colormaps use distinct, non-ordered colours and are "
        "best suited for categorical or labelled data where the category "
        "identity matters more than any ordering.",
        [
            "Accent", "Accent_r",
            "Dark2", "Dark2_r",
            "Paired", "Paired_r",
            "Pastel1", "Pastel1_r",
            "Pastel2", "Pastel2_r",
            "Set1", "Set1_r",
            "Set2", "Set2_r",
            "Set3", "Set3_r",
            "flag", "flag_r",
            "prism", "prism_r",
            "brg", "brg_r",
            "gist_ncar", "gist_ncar_r",
            "gist_rainbow", "gist_rainbow_r",
            "jet", "jet_r",
            "nipy_spectral", "nipy_spectral_r",
            "rainbow", "rainbow_r",
        ],
    ),
    (
        "Pycortex-specific 1D",
        "These colormaps were created specifically for pycortex or are "
        "commonly used in neuroimaging.  The ``J``-series colormaps are "
        "designed to be perceptually uniform.  ``HCP_MMP1`` and "
        "``freesurfer_aseg_256`` are specialised palettes for standard "
        "brain parcellations.",
        [
            "BPROG",
            "BROYG",
            "CyanBlueGrayRedPink",
            "J4", "J4R", "J4s",
            "J5", "J5R",
            "J6", "J6R",
            "HCP_MMP1",
            "freesurfer_aseg_256",
            "spectral", "spectral_r",
        ],
    ),
]

CATEGORIES_2D = [
    (
        "2D Diverging",
        "These 2D colormaps map two independent data dimensions to colour "
        "simultaneously, with each axis using a diverging scheme centred on "
        "a neutral colour.  They are particularly useful for displaying two "
        "complementary contrasts (e.g. mean and variance, or two "
        "experimental conditions) overlaid on the same cortical surface.",
        [
            "BuOr_2D",
            "BuBkRd_alpha_2D",
            "BuWtRd_alpha",
            "BuWtRd_black_2D",
            "GreenWhiteBlue_2D",
            "GreenWhiteRed_2D",
            "PU_BuOr_covar",
            "PU_BuOr_covar_alpha",
            "PU_PinkBlue_covar",
            "PU_RdBu_covar",
            "PU_RdBu_covar_alpha",
            "PU_RdGn_covar",
            "RdBu_2D",
            "RdBu_2D_r",
            "RdBu_covar",
            "RdBu_covar2",
            "RdBu_covar_alpha",
            "RdBu_r_alpha",
            "RdGn_covar",
            "Reds_cov",
        ],
    ),
    (
        "2D Retinotopy",
        "These 2D colormaps encode both polar angle and eccentricity in a "
        "single image and are designed for visualising retinotopic maps on "
        "the cortical surface.",
        [
            "Retinotopy_HSV_alpha",
            "Retinotopy_HSV_2x_alpha",
            "Retinotopy_RYBCR_2D",
            "Retinotopy_RYBCR_alpha",
            "eccentricity_alpha_2D",
        ],
    ),
    (
        "2D Sequential",
        "These 2D colormaps vary from dark to light (or transparent) along "
        "one axis and change hue along the other.  They can be used to show "
        "a primary response (hue axis) together with its confidence or "
        "magnitude (lightness/alpha axis).",
        [
            "autumn_alpha",
            "autumn_blkmin_alpha_2D",
            "autumnblack_alpha_2D",
            "fire_alpha",
            "hot_alpha",
            "nipy_spectral_alpha",
            "plasma_alpha",
            "seismic_alpha",
            "spectral_alpha",
        ],
    ),
    (
        "2D Miscellaneous",
        "Other 2D colormaps that do not fit neatly into the categories "
        "above.",
        [
            "BROYG_2D",
            "custom2D_RB_bins_256",
        ],
    ),
]


def write_cmap_entry(f, name, path, is2d):
    """Write a single colormap image block to the RST file."""
    f.write(f"**{name}**\n\n")
    f.write(f".. image:: {path}\n")
    if is2d:
        f.write("   :height: 200px\n")
        f.write("   :width: 200px\n")
    else:
        f.write("   :height: 25px\n")
        f.write("   :width: 400px\n")
    f.write("\n\n")


def write_section(f, title, description, names, level="~"):
    """Write a titled section with a description and colormap entries."""
    f.write(f"{title}\n")
    f.write(level * len(title) + "\n\n")
    f.write(f"{description}\n\n")

    available = {n[:-4] for n in os.listdir(path_to_colormaps)
                 if n.endswith(".png")}
    for name in names:
        if name not in available:
            continue
        path = os.path.join(path_to_colormaps, name + ".png")
        write_cmap_entry(f, name, path, is_2d(name))


# ---------------------------------------------------------------------------
# Generate colormaps.rst
# ---------------------------------------------------------------------------

with open("colormaps.rst", "w") as rst_file:
    rst_file.write("Colormaps\n")
    rst_file.write("=========\n\n")
    rst_file.write(
        "Pycortex ships with a large collection of colormaps for both "
        "1D (standard) and 2D data.  This page organises them into "
        "meaningful categories, similar to how `matplotlib documents its "
        "colormaps <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.  "
        "Each colormap name can be passed directly "
        "to :func:`cortex.quickflat.make_figure` or "
        ":func:`cortex.webgl.show` via the ``cmap`` argument.\n\n"
        "All 1D colormaps that end in ``_r`` are the reversed version of "
        "the base colormap.  Custom colormaps can be added with "
        ":func:`cortex.utils.add_cmap`.\n\n"
    )

    rst_file.write("1D Colormaps\n")
    rst_file.write("------------\n\n")
    rst_file.write(
        "Standard 1D (linear) colormaps that map a single data dimension "
        "to colour.\n\n"
    )

    for title, description, names in CATEGORIES_1D:
        write_section(rst_file, title, description, names, level="~")

    rst_file.write("2D Colormaps\n")
    rst_file.write("------------\n\n")
    rst_file.write(
        "2D colormaps encode **two independent data dimensions** "
        "simultaneously using colour.  The horizontal axis of the square "
        "swatch represents one data dimension and the vertical axis "
        "represents the other.  These colormaps are used with "
        ":class:`cortex.dataset.Volume2D`.\n\n"
    )

    for title, description, names in CATEGORIES_2D:
        write_section(rst_file, title, description, names, level="~")
