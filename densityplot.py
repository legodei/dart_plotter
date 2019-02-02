# libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
import seaborn as sns

x = nightrifle[:,0]
y = nightrifle[:,1]
 
x2 = final3[:,0]
y2 = final3[:,1]
# library & dataset


# Change shape of marker
#sns.regplot(x, y, marker='o', fit_reg=False)
#sns.plt.show()

# You can see all possible shapes:
#all_shapes=markers.MarkerStyle.markers.keys()
#all_shapes
#[0, 1, 2, 3, 4, u'D', 6, 7, 8, u's', u'|', 11, u'None', u'P', 9, u'x', u'X', 5, u'_', u'^', u' ', None, u'd', u'h', u'+', u'*', u',', u'o', u'.', u'1', u'p', u'3', u'2', u'4', u'H', u'v', u'', u'8', 10, u'&lt;', u'&gt;']

# More marker customization:
ax = plt.figure()
sns.regplot(x, y,marker='s', fit_reg=False, scatter_kws={"color":"red","alpha":0.2,"s":100},label="rifled" )
sns.regplot(x2, y2, marker='D', fit_reg=False, scatter_kws={"color":"green","alpha":0.2,"s":100},label="smoothbore" )
ax.legend()
#sns.plt.show()
