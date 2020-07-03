from sklearn.decomposition import PCA
import numpy as np

# geometry
order = ['area',
         'volume',
         'area_volume_ratio',
         'edge_ratio',
         'radius_ratio',
         'aspect_ratio',
         'max_solid_angle',
         'min_solid_angle',
         'solid_angle']

# Dirichlet Distribution alphas
alphas = np.arange(1, 11, 0.5)

# Compression Methods
projectors = [PCA()]

train_smote_ext = ["_train", "_SMOTE", "_Borderline1", "_Borderline2", "_smoteSVM", "_Geometric_SMOTE"]

datasets = [
	'cal_housing',
	'house_8L',
	'abalone',
	'cpu',
	'house_16H',
	'ailerons',
	'cpu_act',
	'quake',
	'autoPrice',
	'cpu_small',
	'bank8FM',
	'machine_cpu',
	'bank32nh',
	'triazines',
	'elevators',
	'fried',
	'puma32H'
]
