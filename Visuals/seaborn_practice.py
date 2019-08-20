# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:30:22 2019

@author: Will Han
"""
#http://seaborn.pydata.org/tutorial/relational.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")

tips = sns.load_dataset("tips")
tips.head()
plt.scatter(x=tips['total_bill'], y = tips['tip'], data = tips)
plt.scatter(x='total_bill', y = 'tip', data = tips)
sns.relplot(x = "total_bill", y = "tip", data = tips)

sns.relplot(x = 'total_bill', y = 'tip', hue = 'smoker', data = tips)

sns.relplot(x = 'total_bill', y = 'tip', hue = 'smoker', style = 'smoker', data = tips)

sns.relplot(x = 'total_bill', y = 'tip', hue = 'smoker', style = 'time', data = tips)

sns.relplot(x = 'total_bill', y = 'tip', hue = 'size', data = tips)

sns.relplot(x = 'total_bill', y = 'tip', hue = 'size', palette = 'ch:r = -.5,l = .75', data = tips)

sns.relplot(x = 'total_bill', y = 'tip', size = 'size', sizes = (15,200), data = tips)

# Emphasizing continuity with line plots
df = pd.DataFrame(dict(time = np.arange(500),
                      value = np.random.randn(500).cumsum()))
g = sns.relplot(x = 'time', y = 'value', kind = 'line', data = df)
g.fig.autofmt_xdate()

# lineplot() assumes that you are most often trying to draw y as a function of x, the default behavior is to sort the data by the x values before plotting.  This can be disabled:
df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis = 0), columns = ['x', 'y'])
sns.relplot(x = 'x', y = 'y', sort = False, kind = 'line', data = df)

# Aggregation and representing uncertainty
# for multiple measurements for the same values of x, the default behavior in seaborn is to aggregate the multiple measurements at each x value by plotting the mean and the 95% CI around the mean
fmri = sns.load_dataset('fmri')
fmri.head()
sns.relplot(x = 'timepoint', y = 'signal', kind = 'line', data = fmri)
# The CI's are computed using bootstrapping, which can be time-consuming for larger datasets.  It's possible to disable them.
sns.relplot(x = 'timepoint', y = 'signal', ci = None, kind = 'line', data = fmri)
# can represent the spread of the distribution at each timepoint by plotting the standard deviation instead of the CI
sns.relplot(x = 'timepoint', y = 'signal', kind = 'line', ci = 'sd', data = fmri)
# to turn off aggregation altogether, set the estimator parameter to None.
sns.relplot(x = 'timepoint', y = 'signal', estimator = None, kind = 'line', data = fmri)

# Line plots work with the same parameters as scatter plot so we can add additional dimensions with hue, style, and size
sns.relplot(x = 'timepoint', y = 'signal', hue = 'event', kind = 'line', data = fmri)
# add style
sns.relplot(x = 'timepoint', y = 'signal', hue = 'region', style = 'event', kind = 'line', data = fmri)
