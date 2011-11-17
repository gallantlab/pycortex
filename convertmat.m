addpath('/auto/k1/kendrick/utils/matlab/surfpak/');
addpath('/auto/k1/kendrick/utils/matlab/');
addpath('/auto/k1/kendrick/utils/matlab/surefit/');
addpath('/auto/k2/share/kathleen/anatomicals/code/');

files = {
	struct('subject', 'SN', 'name','SN_shinji', 'xfm', '/auto/data/shinji/MRI/voxels/voxelsSN/tr/tr_20100714SN_anatKat.mat'),
	struct('subject', 'JV', 'name','AV_huth', 'xfm','/auto/k1/huth/text/movie/data/20101002AV-tr.mat'),
	struct('subject', 'JV', 'name', 'AV_shinji','xfm','/auto/data/shinji/MRI/voxels/voxelsAV/tr/tr20090116AV_2_anatomical.mat'),
	struct('subject', 'AH', 'name', 'AH_huth','xfm','/auto/k1/huth/text/movie/data/AH_tr.mat'),
	struct('subject', 'AH', 'name', 'AH_shinji','xfm','/auto/data/shinji/MRI/voxels/voxelsAH/tr/tr_20100824AH_anatKat.mat'),
	struct('subject', 'TC', 'name', 'TC_huth','xfm','/auto/k1/huth/text/movie/data/temptr_TC.mat'),
	struct('subject', 'TC', 'name', 'TC_shinji','xfm','/auto/data/shinji/MRI/voxels/voxelsTC/tr/tr_20101001TC_anatKat2.mat'),
	struct('subject', 'NB', 'name', 'NB_huth','xfm','/auto/k1/huth/text/movie/data/temptr_NB.mat'),
	struct('subject', 'WH', 'name', 'WH_huth','xfm','/auto/k1/huth/text/movie/data/temptr_WH.mat'), 
	struct('subject', 'NB', 'name', 'NB_cukur', 'xfm', '/auto/k6/cukur/flatmap/alignments/temptr_NB2.mat')
};

for i=1:length(files)
	load(files{i}.xfm)
	vtk = eval(['brain_' files{i}.subject '_lh(''fiducial'')']);
	[XYZ, polys, tmp] = loadvtk(vtk{:});
	flat = loadvtk(vtk{1}, [0,0,0]);

	out = volumetoslices(XYZ, tr, []);
	out(1,:) = out(1,:) - 1;
	out(2,:) = out(2,:) - 1;
	out(3,:) = out(3,:) - 1;
	mat = (flat'\out')';

	files{i}.xfm = mat;
end
save /tmp/xfmmats.mat files