# libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def centroidnp(arr):
	''' calculates the centroid given an array of points '''
	assert arr.shape[1] == 2
	length = arr.shape[0]
	sum_x = np.sum(arr[:, 0])
	sum_y = np.sum(arr[:, 1])
	return sum_x/length, sum_y/length

def radiusvalue(arr):
	''' returns a list of radiuses given an array of points '''
	assert arr.shape[1] == 2
	x2 = np.power(arr[:,0],2)
	y2 = np.power(arr[:,1],2)
	return np.power(x2+y2,0.5)
	
	
Files = [ # Enter Data files here
'rifle_acc_1',
'rifle_acc_2',
'rifle_wkr_1',
'straight_acc_1',
'straight_acc_2',
'straight_wkr_1'
]

el = np.zeros((0,2),np.int32) # empty list

raw_all = {'rifle_acc':[],'straight_acc':[],'rifle_wkr':[],'straight_wkr':[]} # raw input data
zeroed_all = {'rifle_acc':el,'straight_acc':el,'rifle_wkr':el,'straight_wkr':el}
radius_all = {'rifle_acc':[],'straight_acc':[],'rifle_wkr':[],'straight_wkr':[]} 

for filename in Files: # Loop through imports and sort based off of attachment
	if 'rifle_acc' in filename:
		raw_all['rifle_acc'].append(np.loadtxt(filename+'.csv', delimiter=","))
	if 'straight_acc' in filename:
		raw_all['straight_acc'].append(np.loadtxt(filename+'.csv', delimiter=","))
	if 'rifle_wkr' in filename:
		raw_all['rifle_wkr'].append(np.loadtxt(filename+'.csv', delimiter=","))
	if 'straight_wkr' in filename:
		raw_all['straight_wkr'].append(np.loadtxt(filename+'.csv', delimiter=","))

for muzzle in list(raw_all.keys()): # Loop, normalized coords around centroid
	for samplelist in raw_all[muzzle]:
		cx,cy = centroidnp(samplelist)
		zeroed_list = np.transpose(np.array([samplelist[:,0]-cx,samplelist[:,1]-cy]))
		zeroed_all[muzzle] = np.append(zeroed_all[muzzle],zeroed_list,axis=0)

for muzzle in list(zeroed_all.keys()): # Finds distance to centroid
	radius_list = list(radiusvalue(zeroed_all[muzzle]))
	radius_all[muzzle].extend(radius_list)
		
filler = True
thickness = 1


# KDE Density Curve
plt.figure()
sns.kdeplot(radius_all['straight_wkr'], color="navy", shade = filler, linewidth = thickness, label="straight flutes, Worker")
sns.kdeplot(radius_all['rifle_wkr'], color="dodgerblue", shade = filler, linewidth = thickness,label="rifled flutes, Worker")
sns.kdeplot(radius_all['straight_acc'], color="darkgreen", shade = filler, linewidth = thickness,label="straight flutes ACCg3")
ax = sns.kdeplot(radius_all['rifle_acc'], color="limegreen", shade = filler, linewidth = thickness,label="rifled flutes ACCg3")
leg = plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)
ax.set_xlim(left=0, right=100)
plt.title('Trajectory Distribution Plot')
plt.xlabel('Distance from center (pixels)')
plt.ylabel('Density')
plt.show()


# Cumulative Density Curve
cdfdist = plt.figure()
sns.kdeplot(radius_all['straight_wkr'], cumulative = True, color="navy", shade = filler, linewidth = thickness, label="straight flutes, Worker")
sns.kdeplot(radius_all['rifle_wkr'], cumulative = True, color="dodgerblue", shade = filler, linewidth = thickness,label="rifled flutes, Worker")
sns.kdeplot(radius_all['straight_acc'], cumulative = True, color="darkgreen", shade = filler, linewidth = thickness,label="straight flutes ACCg3")
ax2 = sns.kdeplot(radius_all['rifle_acc'], cumulative = True, color="limegreen", shade = filler, linewidth = thickness,label="rifled flutes ACCg3")
leg = plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)
ax2.set_xlim(left=0, right=100)

sw = sorted(radius_all['straight_wkr'])
rw = sorted(radius_all['rifle_wkr'])
sa = sorted(radius_all['straight_acc'])
ra = sorted(radius_all['rifle_acc'])

plt.scatter(sw[int(len(sw)*0.75)],0.75,color="navy")
plt.scatter(rw[int(len(rw)*0.75)],0.75,color="dodgerblue")
plt.scatter(sa[int(len(sa)*0.75)],0.75,color="darkgreen")
plt.scatter(ra[int(len(ra)*0.75)],0.75,color="limegreen")



plt.title('Cumulative Distribution Plot')
plt.xlabel('Distance from center (pixels)')
plt.ylabel('Density')
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		