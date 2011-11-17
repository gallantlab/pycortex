import os
import sys
import nibabel

epis = dict(
	SN='/auto/k5/huth/docdb/file_store2/6455827449346130842.nii',
	JV="/auto/k5/huth/docdb/file_store/5872048561085486358.nii",
	AH='/auto/k5/huth/docdb/file_store/7490374805293367026.nii',
	TC='/auto/k5/huth/docdb/file_store2/16672676154421108466.nii',
	NB='/auto/k5/huth/docdb/file_store2/1025627311176466626.nii',
	WH='/auto/k5/huth/docdb/file_store2/14091930909189756342.nii'
)

if __name__ == "__main__":
	from db import surfs
	import scipy.io as sio
	import nibabel
	trs = sio.loadmat(sys.argv[1])['files']
	for t in trs:
		s = t[0]['subject'] if t[0]['subject'] is not 'JV' else 'AV'
		args = [t[0]['subject'], t[0]['name'], t[0]['xfm']]
		kwargs = dict(xfmtype='coord', filename=epis[t[0]['subject']], override=True)
		surfs.loadXfm(*args, **kwargs)

		nii = nibabel.load(epis[t[0]['subject']])

		args[-1] = np.dot(nii.get_affine(), t[0]['xfm'])
		kwargs['xfmtype'] = "magnet"
		surfs.loadXfm(*args, **kwargs)