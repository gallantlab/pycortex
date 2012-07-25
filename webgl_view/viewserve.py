import serve

class Mixer(serve.WebApp):
    def __init__(self, data, subj, xfmname, surfs=('inflated',)):
        

def embedData(*args):
    assert all([len(a) == len(args[0]) for a in args])
    assert all([a.dtype == args[0].dtype for a in args])
    shape = (np.ceil(len(args[0]) / 256.), 256)
    outstr = "";
    for data in args:
        if (data.dtype == np.uint8 and data.ndim == 2 and data.shape[1] in [3,4]):
            mm = 0,0
            outmap = np.zeros((np.prod(shape), 4), dtype=np.uint8)
            outmap[:,-1] = 255
            outmap[:len(data),:data.shape[1]] = data
        else:
            outmap = np.zeros(shape, dtype=np.float32)
            mm = data.min(), data.max()
            outmap.ravel()[:len(data)] = (data - data.min()) / (data.max() - data.min())

        outstr += struct.pack('2f', mm[0], mm[1])+outmap.tostring()
        
    return struct.pack('3I', len(args), shape[1], shape[0])+outstr