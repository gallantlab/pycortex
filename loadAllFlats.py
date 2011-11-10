import os

if __name__ == "__main__":
	cwd = os.path.split(os.path.abspath(__file__))[0]
	fpath = os.path.join(cwd, "flats.sql")
	if os.path.exists(fpath):
		os.unlink(fpath)

	from db import flats
	#names = 'AH,AV,DS,JG,ML,NB,SN,TC,WH'.split(',')
	#path = '/home/james/flatmaps/%s/'
	names = "AH,AV,DS,JG,ML,MO,NB,SN,TC,TN,WH".split(",")
	path = "/auto/data/archive/mri_flats/%s/"
	for n in names:
		flats.loadVTKdir(path%n, n)
