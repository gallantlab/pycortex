import os

if __name__ == "__main__":
	cwd = os.path.split(os.path.abspath(__file__))[0]
	fpath = os.path.join(cwd, "flats.hf5")
	if os.path.exists(fpath):
		os.unlink(fpath)

	from db import DBwrite
	names = 'AH,AV,DS,JG,ML,MO,NB,SN,TC,TN,WH'.split(',')
	path = '/auto/data/archive/mri_flats/%s/'
	db = DBwrite()
	for n in names:
		db.loadVTKdir(path%n, n)
	db.commit()