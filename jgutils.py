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