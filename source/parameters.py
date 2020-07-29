from sklearn.decomposition import PCA
import numpy as np


rank_dir = './../results/rank/'
folder_experiments = './../datasets/'
result_dir = './../results/'
output_dir = './../output/media/'
graphics_dir = './../output/graph/'
# geometry
order = [  # 'area',
	# 'volume',
	# 'area_volume_ratio',
	# 'edge_ratio',
	# 'radius_ratio',
	# 'aspect_ratio',
	'max_solid_angle',
	'min_solid_angle',
	'solid_angle']
# Dirichlet Distribution alphas
alphas = np.arange(1, 11, 0.5)
# Compression Methods
projectors = [PCA()]
train_smote_ext = ["_train", "_SMOTE", "_Borderline1", "_Borderline2", "_smoteSVM", "_Geometric_SMOTE"]
regression_measures = ['R2score', 'MAE', 'MSE', 'MAX']
datasets = [
	'abalone',
	'ailerons',
	'cpu_act',
	'autoPrice',
	'cpu_small',
	'bank8FM',
	'bank32nh',
	'elevators',
	'fried',
	'puma32H'
]
