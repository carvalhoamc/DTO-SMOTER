'''
Based on: Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise
https://github.com/nickkunz/smogn
'''

## load dependencies - third party
import numpy as np
import pandas as pd

## synthetic minority over-sampling technique for regression with gaussian noise
from phi import phi
from phi_ctrl_pts import phi_ctrl_pts


def dtosmoter(
		
		## main arguments / inputs
		data,  ## training set (pandas dataframe)
		y,  ## response variable y by name (string)
		
		drop_na_col=False,  ## auto drop columns with nan's (bool)
		drop_na_row=False,  ## auto drop rows with nan's (bool)
		
		## phi relevance function arguments / inputs
		rel_thres=0.5,  ## relevance threshold considered rare (pos real)
		rel_method="auto",  ## relevance method ("auto" or "manual")
		rel_xtrm_type="both",  ## distribution focus ("high", "low", "both")
		rel_coef=1.5,  ## coefficient for box plot (pos real)
		rel_ctrl_pts_rg=None,  ## input for "manual" rel method  (2d array)
		oversampler=None,
):
	"""
	the main function, designed to help solve the problem of imbalanced data
	for regression, much the same as SMOTE for classification; SMOGN applies
	the combintation of under-sampling the majority class (in the case of
	regression, values commonly found near the mean of a normal distribution
	in the response variable y) and over-sampling the minority class (rare
	values in a normal distribution of y, typically found at the tails)

	procedure begins with a series of pre-processing steps, and to ensure no
	missing values (nan's), sorts the values in the response variable y by
	ascending order, and fits a function 'phi' to y, corresponding phi values
	(between 0 and 1) are generated for each value in y, the phi values are
	then used to determine if an observation is either normal or rare by the
	threshold specified in the argument 'rel_thres'

	normal observations are placed into a majority class subset (normal bin)
	and are under-sampled, while rare observations are placed in a seperate
	minority class subset (rare bin) where they're over-sampled

	under-sampling is applied by a random sampling from the normal bin based
	on a calculated percentage control by the argument 'samp_method', if the
	specified input of 'samp_method' is "balance", less under-sampling (and
	over-sampling) is conducted, and if "extreme" is specified more under-
	sampling (and over-sampling is conducted)

	over-sampling is applied one of two ways, either synthetic minority over-
	sampling technique for regression 'smoter' or 'smoter-gn' which applies a
	similar interpolation method to 'smoter', but takes an additional step to
	perturb the interpolated values with gaussian noise

	'smoter' is selected when the distance between a given observation and a
	selected nearest neighbor is within the maximum threshold (half the median
	distance of k nearest neighbors) 'smoter-gn' is selected when a given
	observation and a selected nearest neighbor exceeds that same threshold

	both 'smoter' and 'smoter-gn' are only applied to numeric / continuous
	features, synthetic values found in nominal / categorical features, are
	generated by randomly selecting observed values found within their
	respective feature

	procedure concludes by post-processing and returns a modified pandas data
	frame containing under-sampled and over-sampled (synthetic) observations,
	the distribution of the response variable y should more appropriately
	reflect the minority class areas of interest in y that are under-
	represented in the original training set

	ref:

	Branco, P., Torgo, L., Ribeiro, R. (2017).
	SMOGN: A Pre-Processing Approach for Imbalanced Regression.
	Proceedings of Machine Learning Research, 74:36-50.
	http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
	"""
	
	## pre-process missing values
	if bool(drop_na_col) == True:
		data = data.dropna(axis=1)  ## drop columns with nan's
	
	if bool(drop_na_row) == True:
		data = data.dropna(axis=0)  ## drop rows with nan's
	
	## quality check for missing values in dataframe
	if data.isnull().values.any():
		raise ValueError("cannot proceed: data cannot contain NaN values")
	
	## quality check for y
	if isinstance(y, str) is False:
		raise ValueError("cannot proceed: y must be a string")
	
	if y in data.columns.values is False:
		raise ValueError("cannot proceed: y must be an header name (string) \
               found in the dataframe")
	
	## quality check for relevance threshold parameter
	if rel_thres == None:
		raise ValueError("cannot proceed: relevance threshold required")
	
	if rel_thres > 1 or rel_thres <= 0:
		raise ValueError("rel_thres must be a real number number: 0 < R < 1")
	
	## store data dimensions
	n = len(data)
	d = len(data.columns)
	
	## store original data types
	feat_dtypes_orig = [None] * d
	
	for j in range(d):
		feat_dtypes_orig[j] = data.iloc[:, j].dtype
	
	## determine column position for response variable y
	y_col = data.columns.get_loc(y)
	
	## move response variable y to last column
	if y_col < d - 1:
		cols = list(range(d))
		cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
		data = data[data.columns[cols]]
	
	## store original feature headers and
	## encode feature headers to index position
	feat_names = list(data.columns)
	data.columns = range(d)
	
	## sort response variable y by ascending order
	y = pd.DataFrame(data[d - 1])
	y_sort = y.sort_values(by=d - 1)
	y_sort = y_sort[d - 1]
	
	## -------------------------------- phi --------------------------------- ##
	## calculate parameters for phi relevance function
	## (see 'phi_ctrl_pts()' function for details)
	phi_params = phi_ctrl_pts(
			
			y=y_sort,  ## y (ascending)
			method=rel_method,  ## defaults "auto"
			xtrm_type=rel_xtrm_type,  ## defaults "both"
			coef=rel_coef,  ## defaults 1.5
			ctrl_pts=rel_ctrl_pts_rg  ## user spec
	)
	
	## calculate the phi relevance function
	## (see 'phi()' function for details)
	y_phi = phi(
			
			y=y_sort,  ## y (ascending)
			ctrl_pts=phi_params  ## from 'phi_ctrl_pts()'
	)
	
	## phi relevance quality check
	if all(i == 0 for i in y_phi):
		raise ValueError("redefine phi relevance function: all points are 1")
	
	if all(i == 1 for i in y_phi):
		raise ValueError("redefine phi relevance function: all points are 0")
	## ---------------------------------------------------------------------- ##
	
	## determine bin (rare or normal) by bump classification
	bumps = [0]
	
	for i in range(0, len(y_sort) - 1):
		if ((y_phi[i] >= rel_thres and y_phi[i + 1] < rel_thres) or
				(y_phi[i] < rel_thres and y_phi[i + 1] >= rel_thres)):
			bumps.append(i + 1)
	
	bumps.append(n)
	
	## number of bump classes
	n_bumps = len(bumps) - 1
	
	## determine indicies for each bump classification
	b_index = {}
	
	for i in range(n_bumps):
		b_index.update({i: y_sort[bumps[i]:bumps[i + 1]]})
	
	## conduct over / under sampling and store modified training set
	
	imblearn_Y = np.zeros(n)
	for i in range(n_bumps):
		imblearn_Y[b_index[i].index] = i
	
	data_new = pd.DataFrame(oversampler.fit_resample(data, imblearn_Y)[0])
	
	## restore response variable y to original position
	if y_col < d - 1:
		cols = list(range(d))
		cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
		data_new = data_new[data_new.columns[cols]]
	
	## return modified training set
	return data_new
