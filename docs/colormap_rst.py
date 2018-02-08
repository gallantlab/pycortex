"""
This will make the colormaps.rst file for the docs page.
"""
import os

path_to_colormaps = "../filestore/colormaps/"
all_colormaps = os.listdir(path_to_colormaps)
all_colormaps.sort()
all_paths = [os.path.join(path_to_colormaps, f) for f in all_colormaps]
all_names = [f[:-4] for f in all_colormaps]

rst_lines = ["Colormaps\n",
             "=========\n",
             "\n",
             "There are a number of colormaps available in pycortex.\n"
             "A full list of those that can be used are below.\n"
             "\n"]

rst_file = open("colormaps.rst", "w")
for l in rst_lines:
    rst_file.write(l)

path_template = ".. image:: {path}\n"
width = "   :width: 200px\n"
height_1d = "   :height: 25px\n"
height_2d = "   :height: 200px\n"

for path, name in zip(all_paths, all_names):
    rst_file.write(name)
    rst_file.write("\n\n")
    rst_file.write(path_template.format(path=path))
    if ("2D" in name) or ("covar" in name) or ("alpha" in name):
        rst_file.write(height_2d)
    else:
        rst_file.write(height_1d)
    rst_file.write(width)
    rst_file.write("\n\n")

rst_file.close()
