function parseKathleen(subj)
srcpath = '/auto/k2/share/kathleen/anatomicals/code/';
addpath(srcpath);
types = { 'raw' 'fiducial' 'inflated' 'veryinflated' 'superinflated' 'hyperinflated' 'ellipsoid' 'flat' };
sides = { 'lh', 'rh' };

folder = ['/auto/data/archive/mri_flats/' subj '/'];
mkdir(folder)

for s=1:2
    for t=1:length(types)
        try
            f = eval(['brain_' subj '_' sides{s} '(''' types{t} ''')']);
            copyfile(f{1}, [folder sides{s} '_' types{t} '.vtk']);
        catch err
            fprintf(['blergh, can''t find ' types{t} ': ']);
            fprintf([err.message '\n'])
        end
    end
end
fid = fopen([folder 'coords'], 'w');
fprintf(fid, '%0.3f %0.3f %0.3f', f{2});
fclose(fid);

fid = fopen([srcpath 'brain_' subj '_vol.m'], 'r');
if fid > 0
	contents = fscanf(fid, '%s');
	f = regexp(contents, '''(/.+?\.(nii|img))''', 'tokens');
	fname = f{end}{1};
	copyfile(fname, [folder 'anatomical' fname(end-3:end)]);
end