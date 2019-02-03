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
'metal_1_acc',
'rifle_1_acc',
'rifle_2_acc',
'straight_1',
'straight_2'
]

el = np.zeros((0,2),np.int32) # empty list

raw_all = {'rifle':[],'metal':[],'straight':[]} # raw input data
zeroed_all = {'rifle':el,'metal':el,'straight':el}
radius_all = {'rifle':[],'metal':[],'straight':[]} 

for filename in Files: # Loop through imports and sort based off of attachment
	if 'rifle' in filename:
		raw_all['rifle'].append(np.loadtxt(filename+'.csv', delimiter=","))
	if 'metal' in filename:
		raw_all['metal'].append(np.loadtxt(filename+'.csv', delimiter=","))
	if 'straight' in filename:
		raw_all['straight'].append(np.loadtxt(filename+'.csv', delimiter=","))

for muzzle in list(raw_all.keys()): # Loop, normalized coords around centroid
	for samplelist in raw_all[muzzle]:
		cx,cy = centroidnp(samplelist)
		zeroed_list = np.transpose(np.array([samplelist[:,0]-cx,samplelist[:,1]-cy]))
		zeroed_all[muzzle] = np.append(zeroed_all[muzzle],zeroed_list,axis=0)

for muzzle in list(zeroed_all.keys()): # Finds distance to centroid
	radius_list = list(radiusvalue(zeroed_all[muzzle]))
	radius_all[muzzle].extend(radius_list)
		
sns.distplot(radius_all['metal'], color="blue", label="metal muzzle brake")
sns.distplot(radius_all['straight'], color="red", label="straight flute + muzzle brake")
ax = sns.distplot(radius_all['rifle'], color="green", label="rifled flutes + muzzle brake")
plt.legend()
ax.set_xlim(left=0)
plt.title('Trajectory Distribution Plot')
plt.xlabel('Distance from center (pixels)')
plt.ylabel('Density')
plt.show()


		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		