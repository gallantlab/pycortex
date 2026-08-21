import cortex
from cortex.quickflat import laminar

subject = "S1"
xfmname = "fullhead"

uv_l = [100, -210]
uv_r = [50, -150]
W, H = 500, 100

laminar_profile = laminar.make_laminar_profile(subject, xfmname, uv_l[0], uv_l[1], uv_r[0], uv_r[1], W, H)
