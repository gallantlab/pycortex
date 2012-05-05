import numpy as np

def unmask(mask, data):
    '''unmask(mask, data)

    "Unmasks" the data, assuming it's been masked.

    Parameters
    ----------
    mask : array_like or str or docdb.orm.ImageDoc
        The data mask -- if string, assume it's the experiment name and query the
        BrainMaskFSL document. Otherwise, stuff data into mask
    data : array_lie
        Actual MRI data to unmask
    '''
    #assert len(data.shape) == 2, "Are you sure this is masked data?"
    import docdb
    if isinstance(mask, str):
        client = docdb.getclient()
        mask = client.query(experiment_name=mask, generated_by_name="BrainMaskFSL")[0]

    if isinstance(mask, docdb.orm.ImageDoc):
        mask = mask.get_data()[:]

    output = np.zeros_like(mask)
    output[mask > 0] = data
    return output

def detrend_volume_poly(data, polyorder = 10, mask=None):
    from scipy.special import legendre
    polys = [legendre(i) for i in range(polyorder)]
    s = data.shape
    b = data.ravel()[:,np.newaxis]
    lins = np.mgrid[-1:1:s[0]*1j, -1:1:s[1]*1j, -1:1:s[2]*1j].reshape(3,-1)

    if mask is not None:
        lins = lins[:,mask.ravel() > 0]
        b = b[mask.ravel() > 0]
    
    A = np.vstack([[p(i) for i in lins] for p in polys]).T
    x, res, rank, sing = np.linalg.lstsq(A, b)

    detrended = b.ravel() - np.dot(A, x).ravel()
    if mask is not None:
        filled = np.zeros_like(mask)
        filled[mask > 0] = detrended
        return filled
    else:
        return detrended.reshape(*s)
