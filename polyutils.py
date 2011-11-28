import numpy as np
from scipy.spatial import distance

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import patches

def get_connected(polys):
	data = [set([]) for _ in range(polys.max()+1)]
	for i, poly in enumerate(polys):
		for p in poly:
			data[p].add(i)

	return data

def check_cycle(i, polys, ptpoly):
	pts = polys[list(ptpoly[i])]
	cycles = pts[np.nonzero(pts-i)]
	return cycles

def remove_pairs(arr):
	return [p for p in np.unique(arr) if sum(p == arr) != 2]

def trace_edge(seed, pts, polys, ptpoly):
	edge = [seed]
	while True:
		cycle = remove_pairs(check_cycle(edge[-1], polys, ptpoly))
		if cycle[0] not in edge:
			edge.append(cycle[0])
		elif cycle[1] not in edge:
			edge.append(cycle[1])
		else:
			#both in edges, we've gone around!
			break;
	return edge

def trace_both(pts, polys):
	ptpoly = get_connected(polys)
	left = trace_edge(pts.argmin(0)[0], pts, polys, ptpoly)
	right = trace_edge(pts.argmax(0)[0], pts, polys, ptpoly)
	return left, right

def get_dist(pts):
	return distance.squareform(distance.pdist(pts))

def get_closest_nonadj(dist, adjthres=10):
	closest = []
	for i, d in enumerate(dist):
		sort = d.argsort()[:adjthres] - i
		sort = np.array([sort, abs(sort - len(dist))]).min(0)
		find = sort[sort > adjthres]
		if len(find) > 0:
			closest.append((i, find[0]+i))

	return np.array(filter(lambda x: (x[1], x[0]) not in closest, closest))

def make_rot(close, pts):
	pair = pts[close, :2]
	refpt = pair[0] - pair[1]
	d1 = close[1] - close[0]
	d2 = len(pts) - close[1] + close[0]
	if d2 < d1:
		refpt = pair[1] - pair[0]
	a = np.arctan2(refpt[1], refpt[0])
	m = np.array([[np.cos(-a), -np.sin(-a)],
				  [np.sin(-a),  np.cos(-a)]])
	return m, d2 < d1

def get_height(closest, pts):
	data = []
	for close in closest:
		m, flip = make_rot(close, pts)
		npt = (pts-pts[close[1]])
		tpt = np.dot(m, npt.T[:2]).T
		if flip:
			data.append(np.vstack([tpt[max(close):], tpt[:min(close)]]).max(0)[1])
		else:
			data.append(tpt[min(close):max(close)].max(0)[1])
	return np.array(data)
	#return np.array([np.dot(make_rot(pts[close]), (pts-pts[close[1]]).T[:2]).max(1)[1] for close in closest])

def get_perp(pts, i, width=11):
	idx = (np.arange(-(width/2), width/2+1)+i) % len(pts)
	x = np.vstack([pts[idx,0], np.ones((width,))]).T
	m, b = np.linalg.lstsq(x, pts[idx, 1])[0]
	a = np.arctan2(-1./m, 1)

	func = lambda h: (np.cos(a)*h + pts[i,0], np.sin(a)*h + pts[i,1])
	p1 = pts[(i+1)%len(pts)] - pts[i]
	p2 = func(1) - pts[i, :2]
	if np.cross(p1[:2], p2) < 0:
		return func
	else:
		return lambda h: func(-h)

def get_ctrl_pts(closest, pts, factor=1.5):
	height = get_height(closest, pts)
	height /= height.max()
	pm = pts.mean(0)[:2]
	cpts = pts[:,:2] - pm
	data = []
	#return [cpts[close]*h*factor + pm for h, close in zip(height, closest)]
	for h, close in zip(height, closest):
		data.append(pts[close[0],:2])
		data.append(get_perp(pts, close[0])((h+1)*factor))
		data.append(get_perp(pts, close[1])((h+1)*factor))
		data.append(pts[close[1],:2])
	return data 

def _test_inside(close, pts):
	ref = pts[close[0]]
	return np.cross(pts[close[0]-1] - ref, pts[close[1]] - ref) > 0

def draw_curves(closest, pts, factor=0.5):
	verts = get_ctrl_pts(closest, pts, factor)
	codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
	path = Path(verts, codes*len(closest))
	patch = patches.PathPatch(path, lw=1, facecolor='none')
	ax = plt.gca()
	ax.add_patch(patch)
	ax.plot(pts.T[0], pts.T[1], 'x-')

def draw_lines(closest, pts):
	inside, outside = [], []
	for close in closest:
		if _test_inside(close, pts[...,:2]):
			inside.append(pts[close, :2])
		else:
			outside.append(pts[close, :2])
	
	codepair = [Path.MOVETO, Path.LINETO]
	inside = Path(np.vstack(inside), codepair*len(inside))
	outside = Path(np.vstack(outside), codepair*len(outside))
	ipatch = patches.PathPatch(inside, lw=1, color='red', facecolor='none')
	opatch = patches.PathPatch(outside, lw=1, color='blue', facecolor='none')

	ax = plt.gca()
	ax.add_patch(ipatch)
	#ax.add_patch(opatch)
	ax.plot(pts.T[0], pts.T[1], 'x-')
	'''
	pts = np.vstack(pairs[...,:2])
	codes = [Path.MOVETO, Path.LINETO]*(len(pts)/2)
	path = Path(pts, codes)
	patch = patches.PathPatch(path, lw=2, facecolor='none')
	ax = plt.gca()
	ax.add_patch(patch)
	'''

if __name__ == "__main__":
	import cPickle
	pts, polys, fpts = cPickle.load(open("/tmp/ptspolys.pkl"))
	left, right = trace_both(pts, polys)
	dist = get_dist(fpts[left])
	rdist = get_dist(fpts[right])
	closest = get_closest_nonadj(dist)
	rclosest = get_closest_nonadj(dist)[:,::-1]

	pairs = pts[left][closest]