import numpy as np
import cortex
from cortex.quickflat import laminar


subject = "S1"
xfmname = "identity"

uv_l = [100, -210]
uv_r = [50, -150]
W, H = 100, 5

laminar_profile = laminar.make_laminar_profile(subject, xfmname, uv_l[0], uv_l[1], uv_r[0], uv_r[1], W, H)

fake_data = np.linspace(200, 500, H)[:,None] @ np.ones(W)[None,:]
anatdata = cortex.db.get_anat(subject).get_fdata().T
fake_vol = anatdata.copy()
fake_vol.ravel()[laminar_profile] += fake_data
fake_vol[fake_vol == 0] = np.nan

cortex.webshow((fake_vol, subject, xfmname))


laminar_data = anatdata.ravel()[laminar_profile]
import matplotlib.pyplot as plt
plt.matshow(laminar_data)