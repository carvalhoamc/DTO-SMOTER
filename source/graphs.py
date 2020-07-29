from os import listdir
from os.path import isfile, join

import Orange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from parameters import order, alphas, regression_measures, datasets, rank_dir, output_dir, graphics_dir
from regression_algorithms import regression_list

results_dir = './../results/'


class Performance:
	
	def __init__(self):
		pass
	
	def average_results(self, rfile, release):
		'''
		Calculates average results
		:param rfile: filename with results
		:param kind: biclass or multiclass
		:return: avarege_results in another file
		'''
		
		df = pd.read_csv(rfile)
		t = pd.Series(data=np.arange(0, df.shape[0], 1))
		dfr = pd.DataFrame(columns=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM', 'ORDER',
		                            'ALPHA', 'R2score', 'MAE', 'MSE', 'MAX'],
		                   index=np.arange(0, int(t.shape[0] / 5)))
		
		df_temp = df.groupby(by=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM'])
		idx = dfr.index.values
		i = idx[0]
		for name, group in df_temp:
			group = group.reset_index()
			dfr.at[i, 'MODE'] = group.loc[0, 'MODE']
			dfr.at[i, 'DATASET'] = group.loc[0, 'DATASET']
			dfr.at[i, 'PREPROC'] = group.loc[0, 'PREPROC']
			dfr.at[i, 'ALGORITHM'] = group.loc[0, 'ALGORITHM']
			dfr.at[i, 'ORDER'] = group.loc[0, 'ORDER']
			dfr.at[i, 'ALPHA'] = group.loc[0, 'ALPHA']
			dfr.at[i, 'R2score'] = group['R2score'].mean()
			dfr.at[i, 'MAE'] = group['MAE'].mean()
			dfr.at[i, 'MSE'] = group['MSE'].mean()
			dfr.at[i, 'MAX'] = group['MAX'].mean()
			i = i + 1
		
		print('Total lines in a file: ', i)
		dfr.to_csv(results_dir + 'regression_average_results_' + str(release) + '.csv', index=False)
	
	def run_rank_choose_parameters(self, filename, release):
		df_best_dto = pd.read_csv(filename)
		df_B1 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline1'].copy()
		df_B2 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline2'].copy()
		df_GEO = df_best_dto[df_best_dto['PREPROC'] == '_Geometric_SMOTE'].copy()
		df_SMOTE = df_best_dto[df_best_dto['PREPROC'] == '_SMOTE'].copy()
		df_SMOTEsvm = df_best_dto[df_best_dto['PREPROC'] == '_smoteSVM'].copy()
		df_original = df_best_dto[df_best_dto['PREPROC'] == '_train'].copy()
		
		for o in order:
			for a in alphas:
				GEOMETRY = '_dto_smoter_' + o + '_' + str(a)
				df_dto = df_best_dto[df_best_dto['PREPROC'] == GEOMETRY].copy()
				df = pd.concat([df_B1, df_B2, df_GEO, df_SMOTE, df_SMOTEsvm, df_original, df_dto])
				self.rank_by_algorithm(df, o, str(a), release)
				self.rank_dto_by(o + '_' + str(a), release)
	
	def rank_by_algorithm(self, df, order, alpha, release, smote=False):
		'''
		Calcula rank
		:param df:
		:param tipo:
		:param wd:
		:param GEOMETRY:
		:return:
		'''
		df_table = pd.DataFrame(
				columns=['DATASET', 'ALGORITHM', 'ORIGINAL', 'RANK_ORIGINAL', 'SMOTE', 'RANK_SMOTE', 'SMOTE_SVM',
				         'RANK_SMOTE_SVM', 'BORDERLINE1', 'RANK_BORDERLINE1', 'BORDERLINE2', 'RANK_BORDERLINE2',
				         'GEOMETRIC_SMOTE', 'RANK_GEOMETRIC_SMOTE', 'DTO', 'RANK_DTO', 'GEOMETRY',
				         'ALPHA', 'unit'])
		
		df_temp = df.groupby(by=['ALGORITHM'])
		for name, group in df_temp:
			
			group = group.reset_index()
			group.drop('index', axis=1, inplace=True)
			if smote == False:
				df.to_csv(rank_dir + release + '_' + order + '_' + str(alpha) + '.csv', index=False)
			else:
				df.to_csv(rank_dir + release + '_smote_' + order + '_' + str(alpha) + '.csv', index=False)
			
			j = 0
			measures = regression_measures
			
			for d in datasets:
				for m in measures:
					aux = group[group['DATASET'] == d]
					aux = aux.reset_index()
					df_table.at[j, 'DATASET'] = d
					df_table.at[j, 'ALGORITHM'] = name
					indice = aux.PREPROC[aux.PREPROC == '_train'].index.tolist()[0]
					df_table.at[j, 'ORIGINAL'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_SMOTE'].index.tolist()[0]
					df_table.at[j, 'SMOTE'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_smoteSVM'].index.tolist()[0]
					df_table.at[j, 'SMOTE_SVM'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_Borderline1'].index.tolist()[0]
					df_table.at[j, 'BORDERLINE1'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_Borderline2'].index.tolist()[0]
					df_table.at[j, 'BORDERLINE2'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_Geometric_SMOTE'].index.tolist()[0]
					df_table.at[j, 'GEOMETRIC_SMOTE'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.ORDER == order].index.tolist()[0]
					df_table.at[j, 'DTO'] = aux.at[indice, m]
					df_table.at[j, 'GEOMETRY'] = order
					df_table.at[j, 'ALPHA'] = alpha
					df_table.at[j, 'unit'] = m
					j += 1
			
			df_r2 = df_table[df_table['unit'] == 'R2score']
			df_mae = df_table[df_table['unit'] == 'MAE']
			df_mse = df_table[df_table['unit'] == 'MSE']
			df_max = df_table[df_table['unit'] == 'MAX']
			
			r2 = df_r2[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DTO']]
			mae = df_mae[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DTO']]
			mse = df_mse[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DTO']]
			max = df_max[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DTO']]
			
			r2 = r2.reset_index()
			r2.drop('index', axis=1, inplace=True)
			mae = mae.reset_index()
			mae.drop('index', axis=1, inplace=True)
			mse = mse.reset_index()
			mse.drop('index', axis=1, inplace=True)
			max = max.reset_index()
			max.drop('index', axis=1, inplace=True)
			
			# calcula rank linha a linha
			r2_rank = r2.rank(axis=1, ascending=False)
			mae_rank = mae.rank(axis=1, ascending=True)
			mse_rank = mse.rank(axis=1, ascending=True)
			max_rank = max.rank(axis=1, ascending=True)
			
			df_r2 = df_r2.reset_index()
			df_r2.drop('index', axis=1, inplace=True)
			df_r2['RANK_ORIGINAL'] = r2_rank['ORIGINAL']
			df_r2['RANK_SMOTE'] = r2_rank['SMOTE']
			df_r2['RANK_SMOTE_SVM'] = r2_rank['SMOTE_SVM']
			df_r2['RANK_BORDERLINE1'] = r2_rank['BORDERLINE1']
			df_r2['RANK_BORDERLINE2'] = r2_rank['BORDERLINE2']
			df_r2['RANK_GEOMETRIC_SMOTE'] = r2_rank['GEOMETRIC_SMOTE']
			df_r2['RANK_DTO'] = r2_rank['DTO']
			
			df_mae = df_mae.reset_index()
			df_mae.drop('index', axis=1, inplace=True)
			df_mae['RANK_ORIGINAL'] = mae_rank['ORIGINAL']
			df_mae['RANK_SMOTE'] = mae_rank['SMOTE']
			df_mae['RANK_SMOTE_SVM'] = mae_rank['SMOTE_SVM']
			df_mae['RANK_BORDERLINE1'] = mae_rank['BORDERLINE1']
			df_mae['RANK_BORDERLINE2'] = mae_rank['BORDERLINE2']
			df_mae['RANK_GEOMETRIC_SMOTE'] = mae_rank['GEOMETRIC_SMOTE']
			df_mae['RANK_DTO'] = mae_rank['DTO']
			
			df_mse = df_mse.reset_index()
			df_mse.drop('index', axis=1, inplace=True)
			df_mse['RANK_ORIGINAL'] = mse_rank['ORIGINAL']
			df_mse['RANK_SMOTE'] = mse_rank['SMOTE']
			df_mse['RANK_SMOTE_SVM'] = mse_rank['SMOTE_SVM']
			df_mse['RANK_BORDERLINE1'] = mse_rank['BORDERLINE1']
			df_mse['RANK_BORDERLINE2'] = mse_rank['BORDERLINE2']
			df_mse['RANK_GEOMETRIC_SMOTE'] = mse_rank['GEOMETRIC_SMOTE']
			df_mse['RANK_DTO'] = mse_rank['DTO']
			
			df_max = df_max.reset_index()
			df_max.drop('index', axis=1, inplace=True)
			df_max['RANK_ORIGINAL'] = max_rank['ORIGINAL']
			df_max['RANK_SMOTE'] = max_rank['SMOTE']
			df_max['RANK_SMOTE_SVM'] = max_rank['SMOTE_SVM']
			df_max['RANK_BORDERLINE1'] = max_rank['BORDERLINE1']
			df_max['RANK_BORDERLINE2'] = max_rank['BORDERLINE2']
			df_max['RANK_GEOMETRIC_SMOTE'] = max_rank['GEOMETRIC_SMOTE']
			df_max['RANK_DTO'] = max_rank['DTO']
			
			# avarege rank
			media_r2_rank = r2_rank.mean(axis=0)
			media_mae_rank = mae_rank.mean(axis=0)
			media_mse_rank = mse_rank.mean(axis=0)
			media_max_rank = max_rank.mean(axis=0)
			
			media_r2_rank_file = media_r2_rank.reset_index()
			media_r2_rank_file = media_r2_rank_file.sort_values(by=0)
			
			media_mae_rank_file = media_mae_rank.reset_index()
			media_mae_rank_file = media_mae_rank_file.sort_values(by=0)
			
			media_mse_rank_file = media_mse_rank.reset_index()
			media_mse_rank_file = media_mse_rank_file.sort_values(by=0)
			
			media_max_rank_file = media_max_rank.reset_index()
			media_max_rank_file = media_max_rank_file.sort_values(by=0)
			
			if smote == False:
				
				# Grava arquivos importantes
				df_r2.to_csv(
						rank_dir + release + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_r2.csv', index=False)
				df_mae.to_csv(
						rank_dir + release + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_mae.csv', index=False)
				df_mse.to_csv(
						rank_dir + release + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_mse.csv', index=False)
				df_max.to_csv(
						rank_dir + release + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_max.csv', index=False)
				
				media_r2_rank_file.to_csv(
						rank_dir + release + '_' + 'media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_r2.csv',
						index=False)
				media_mae_rank_file.to_csv(
						rank_dir + release + '_media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_mae.csv',
						index=False)
				media_mse_rank_file.to_csv(
						rank_dir + release + '_media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_mse.csv',
						index=False)
				media_max_rank_file.to_csv(
						rank_dir + release + '_media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_max.csv',
						index=False)
				
				GEOMETRY = order + '_' + str(alpha)
				
				# grafico CD
				identificadores = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE',
				                   'DTO']
				avranks = list(media_r2_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_' + '_' + GEOMETRY + '_' + name + '_r2.pdf')
				plt.close()
				
				avranks = list(media_mae_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_' + '_' + GEOMETRY + '_' + name + '_mae.pdf')
				plt.close()
				
				avranks = list(media_mse_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_' + '_' + GEOMETRY + '_' + name + '_mse.pdf')
				plt.close()
				
				avranks = list(media_max_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(rank_dir + release + 'cd_' + '_' + GEOMETRY + '_' + name + '_max.pdf')
				plt.close()
				
				print('Delaunay Type= ', GEOMETRY)
				print('Algorithm= ', name)
			
			
			else:
				# Grava arquivos importantes
				df_r2.to_csv(
						rank_dir + release + '_smote_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_r2.csv', index=False)
				df_mae.to_csv(
						rank_dir + release + '_smote_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_mae.csv', index=False)
				df_mse.to_csv(
						rank_dir + release + '_smote_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_mse.csv', index=False)
				df_max.to_csv(
						rank_dir + release + '_smote_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_max.csv', index=False)
				
				media_r2_rank_file.to_csv(
						rank_dir + release + '_smote_media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_r2.csv',
						index=False)
				media_mae_rank_file.to_csv(
						rank_dir + release + '_smote__media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_mae.csv',
						index=False)
				media_mse_rank_file.to_csv(
						rank_dir + release + 'smote__media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_mse.csv',
						index=False)
				media_max_rank_file.to_csv(
						rank_dir + release + 'smote__media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_max.csv',
						index=False)
				
				GEOMETRY = order + '_' + str(alpha)
				
				# grafico CD
				identificadores = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE',
				                   GEOMETRY]
				avranks = list(media_r2_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_smote' + '_' + GEOMETRY + '_' + name + '_pre.pdf')
				plt.close()
				
				avranks = list(media_mae_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_smote' + '_' + GEOMETRY + '_' + name + '_rec.pdf')
				plt.close()
				
				avranks = list(media_mse_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_smote' + '_' + GEOMETRY + '_' + name + '_spe.pdf')
				plt.close()
				
				avranks = list(media_max_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(rank_dir + release + 'cd_smote' + '_' + GEOMETRY + '_' + name + '_f1.pdf')
				plt.close()
				
				print('SMOTE Delaunay Type= ', GEOMETRY)
				print('SMOTE Algorithm= ', name)
	
	def rank_dto_by(self, geometry, release, smote=False):
		
		M = ['_r2.csv', '_mae.csv', '_mse.csv', '_max.csv']
		df_media_rank = pd.DataFrame(columns=['ALGORITHM', 'RANK_ORIGINAL', 'RANK_SMOTE',
		                                      'RANK_SMOTE_SVM', 'RANK_BORDERLINE1', 'RANK_BORDERLINE2',
		                                      'RANK_GEOMETRIC_SMOTE', 'RANK_DTO', 'unit'])
		
		if smote == False:
			name = rank_dir + release + '_total_rank_' + geometry + '_'
		else:
			name = rank_dir + release + '_smote_total_rank_' + geometry + '_'
		
		for m in M:
			i = 0
			for c in regression_list:
				df = pd.read_csv(name + c + m)
				rank_original = df.RANK_ORIGINAL.mean()
				rank_smote = df.RANK_SMOTE.mean()
				rank_smote_svm = df.RANK_SMOTE_SVM.mean()
				rank_b1 = df.RANK_BORDERLINE1.mean()
				rank_b2 = df.RANK_BORDERLINE2.mean()
				rank_geo_smote = df.RANK_GEOMETRIC_SMOTE.mean()
				rank_dto = df.RANK_DTO.mean()
				df_media_rank.loc[i, 'ALGORITHM'] = df.loc[0, 'ALGORITHM']
				df_media_rank.loc[i, 'RANK_ORIGINAL'] = rank_original
				df_media_rank.loc[i, 'RANK_SMOTE'] = rank_smote
				df_media_rank.loc[i, 'RANK_SMOTE_SVM'] = rank_smote_svm
				df_media_rank.loc[i, 'RANK_BORDERLINE1'] = rank_b1
				df_media_rank.loc[i, 'RANK_BORDERLINE2'] = rank_b2
				df_media_rank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = rank_geo_smote
				df_media_rank.loc[i, 'RANK_DTO'] = rank_dto
				df_media_rank.loc[i, 'unit'] = df.loc[0, 'unit']
				i += 1
			
			dfmediarank = df_media_rank.copy()
			dfmediarank = dfmediarank.sort_values('RANK_DTO')
			
			dfmediarank.loc[i, 'ALGORITHM'] = 'avarage'
			dfmediarank.loc[i, 'RANK_ORIGINAL'] = df_media_rank['RANK_ORIGINAL'].mean()
			dfmediarank.loc[i, 'RANK_SMOTE'] = df_media_rank['RANK_SMOTE'].mean()
			dfmediarank.loc[i, 'RANK_SMOTE_SVM'] = df_media_rank['RANK_SMOTE_SVM'].mean()
			dfmediarank.loc[i, 'RANK_BORDERLINE1'] = df_media_rank['RANK_BORDERLINE1'].mean()
			dfmediarank.loc[i, 'RANK_BORDERLINE2'] = df_media_rank['RANK_BORDERLINE2'].mean()
			dfmediarank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = df_media_rank['RANK_GEOMETRIC_SMOTE'].mean()
			dfmediarank.loc[i, 'RANK_DTO'] = df_media_rank['RANK_DTO'].mean()
			dfmediarank.loc[i, 'unit'] = df.loc[0, 'unit']
			i += 1
			dfmediarank.loc[i, 'ALGORITHM'] = 'std'
			dfmediarank.loc[i, 'RANK_ORIGINAL'] = df_media_rank['RANK_ORIGINAL'].std()
			dfmediarank.loc[i, 'RANK_SMOTE'] = df_media_rank['RANK_SMOTE'].std()
			dfmediarank.loc[i, 'RANK_SMOTE_SVM'] = df_media_rank['RANK_SMOTE_SVM'].std()
			dfmediarank.loc[i, 'RANK_BORDERLINE1'] = df_media_rank['RANK_BORDERLINE1'].std()
			dfmediarank.loc[i, 'RANK_BORDERLINE2'] = df_media_rank['RANK_BORDERLINE2'].std()
			dfmediarank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = df_media_rank['RANK_GEOMETRIC_SMOTE'].std()
			dfmediarank.loc[i, 'RANK_DTO'] = df_media_rank['RANK_DTO'].std()
			dfmediarank.loc[i, 'unit'] = df.loc[0, 'unit']
			
			dfmediarank['RANK_ORIGINAL'] = pd.to_numeric(dfmediarank['RANK_ORIGINAL'], downcast="float").round(2)
			dfmediarank['RANK_SMOTE'] = pd.to_numeric(dfmediarank['RANK_SMOTE'], downcast="float").round(2)
			dfmediarank['RANK_SMOTE_SVM'] = pd.to_numeric(dfmediarank['RANK_SMOTE_SVM'], downcast="float").round(2)
			dfmediarank['RANK_BORDERLINE1'] = pd.to_numeric(dfmediarank['RANK_BORDERLINE1'], downcast="float").round(2)
			dfmediarank['RANK_BORDERLINE2'] = pd.to_numeric(dfmediarank['RANK_BORDERLINE2'], downcast="float").round(2)
			dfmediarank['RANK_GEOMETRIC_SMOTE'] = pd.to_numeric(dfmediarank['RANK_GEOMETRIC_SMOTE'],
			                                                    downcast="float").round(2)
			dfmediarank['RANK_DTO'] = pd.to_numeric(dfmediarank['RANK_DTO'], downcast="float").round(2)
			
			if smote == False:
				dfmediarank.to_csv(output_dir + release + '_results_media_rank_' + geometry + m,
				                   index=False)
			else:
				dfmediarank.to_csv(output_dir + release + '_smote_results_media_rank_' + geometry + m,
				                   index=False)
	
	def grafico_variacao_alpha(self, release):
		M = ['_r2', '_mae', '_mse', '_max']
		
		df_alpha_variations_rank = pd.DataFrame()
		df_alpha_variations_rank['alphas'] = alphas
		df_alpha_variations_rank.index = alphas
		
		df_alpha_all = pd.DataFrame()
		df_alpha_all['alphas'] = alphas
		df_alpha_all.index = alphas
		
		for m in M:
			for o in order:
				for a in alphas:
					filename = output_dir + release + '_results_media_rank_' + o + '_' + str(
							a) + m + '.csv'
					print(filename)
					df = pd.read_csv(filename)
					mean = df.loc[8, 'RANK_DTO']
					df_alpha_variations_rank.loc[a, 'AVARAGE_RANK'] = mean

				if m == '_r2':
					measure = 'R2'
				if m == '_mae':
					measure = 'MAE'
				if m == '_mse':
					measure = 'MSE'
				if m == '_max':
					measure = 'MAX'
				
				df_alpha_all[o + '_' + measure] = df_alpha_variations_rank['AVARAGE_RANK'].copy()
				
				fig, ax = plt.subplots()
				ax.set_title('DTO AVARAGE RANK\n ' + 'GEOMETRY = ' + o + '\nMEASURE = ' + measure, fontsize=10)
				ax.set_xlabel('Alpha')
				ax.set_ylabel('Rank')
				ax.plot(df_alpha_variations_rank['AVARAGE_RANK'], marker='d', label='Avarage Rank')
				ax.legend(loc="upper right")
				plt.xticks(range(11))
				fig.savefig(graphics_dir + release + '_pic_' + o + '_' + measure + '.png', dpi=125)
				plt.show()
				plt.close()
		
		# figure(num=None, figsize=(10, 10), dpi=800, facecolor='w', edgecolor='k')
		
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = R2', fontsize=5)
		ax.set_xlabel('Alpha')
		ax.set_ylabel('Rank')
		t1 = df_alpha_all['alphas']
		t2 = df_alpha_all['alphas']
		t3 = df_alpha_all['alphas']
		ft1 = df_alpha_all['max_solid_angle_R2']
		ft2 = df_alpha_all['min_solid_angle_R2']
		ft3 = df_alpha_all['solid_angle_R2']
		ax.plot(t1, ft1, color='tab:brown', marker='o', label='max_solid_angle')
		ax.plot(t2, ft2, color='tab:pink', marker='o', label='min_solid_angle')
		ax.plot(t3, ft3, color='tab:gray', marker='o', label='solid_angle')
		leg = ax.legend(loc='upper right')
		leg.get_frame().set_alpha(0.5)
		plt.xticks(range(12))
		plt.savefig(graphics_dir + release + '_pic_all_r2.png', dpi=800)
		plt.show()
		plt.close()
		df_alpha_all.to_csv(graphics_dir + release + '_pic_all_r2.csv', index=False)
		
		###################
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = MAE', fontsize=5)
		ax.set_xlabel('Alpha')
		ax.set_ylabel('Rank')
		t1 = df_alpha_all['alphas']
		t2 = df_alpha_all['alphas']
		t3 = df_alpha_all['alphas']
		ft1 = df_alpha_all['max_solid_angle_MAE']
		ft2 = df_alpha_all['min_solid_angle_MAE']
		ft3 = df_alpha_all['solid_angle_MAE']
		ax.plot(t1, ft1, color='tab:brown', marker='o', label='max_solid_angle')
		ax.plot(t2, ft2, color='tab:pink', marker='o', label='min_solid_angle')
		ax.plot(t3, ft3, color='tab:gray', marker='o', label='solid_angle')
		leg = ax.legend(loc='upper right')
		leg.get_frame().set_alpha(0.5)
		plt.xticks(range(12))
		plt.savefig(graphics_dir + release + '_pic_all_mae.png', dpi=800)
		plt.show()
		plt.close()
		df_alpha_all.to_csv(graphics_dir + release + '_pic_all_mae.csv', index=False)
		
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = MSE', fontsize=5)
		ax.set_xlabel('Alpha')
		ax.set_ylabel('Rank')
		t1 = df_alpha_all['alphas']
		t2 = df_alpha_all['alphas']
		t3 = df_alpha_all['alphas']
		ft1 = df_alpha_all['max_solid_angle_MSE']
		ft2 = df_alpha_all['min_solid_angle_MSE']
		ft3 = df_alpha_all['solid_angle_MSE']
		ax.plot(t1, ft1, color='tab:brown', marker='o', label='max_solid_angle')
		ax.plot(t2, ft2, color='tab:pink', marker='o', label='min_solid_angle')
		ax.plot(t3, ft3, color='tab:gray', marker='o', label='solid_angle')
		leg = ax.legend(loc='upper right')
		leg.get_frame().set_alpha(0.5)
		plt.xticks(range(12))
		plt.savefig(graphics_dir + release + '_pic_all_mse.png', dpi=800)
		plt.show()
		plt.close()
		df_alpha_all.to_csv(graphics_dir + release + '_pic_all_mse.csv', index=False)
		
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = MAX', fontsize=5)
		ax.set_xlabel('Alpha')
		ax.set_ylabel('Rank')
		t1 = df_alpha_all['alphas']
		t2 = df_alpha_all['alphas']
		t3 = df_alpha_all['alphas']
		ft1 = df_alpha_all['max_solid_angle_MAX']
		ft2 = df_alpha_all['min_solid_angle_MAX']
		ft3 = df_alpha_all['solid_angle_MAX']
		ax.plot(t1, ft1, color='tab:brown', marker='o', label='max_solid_angle')
		ax.plot(t2, ft2, color='tab:pink', marker='o', label='min_solid_angle')
		ax.plot(t3, ft3, color='tab:gray', marker='o', label='solid_angle')
		leg = ax.legend(loc='upper right')
		leg.get_frame().set_alpha(0.5)
		plt.xticks(range(12))
		plt.savefig(graphics_dir + release + '_pic_all_max.png', dpi=800)
		plt.show()
		plt.close()
		df_alpha_all.to_csv(graphics_dir + release + '_pic_all_max.csv', index=False)
		
	
	def best_alpha(self, kind):
		# Best alpha calculation
		# GEO
		df1 = pd.read_csv(output_dir + 'v1' + '_pic_all_geo.csv')
		df2 = pd.read_csv(output_dir + 'v2' + '_pic_all_geo.csv')
		df3 = pd.read_csv(output_dir + 'v3' + '_pic_all_geo.csv')
		
		if kind == 'biclass':
			col = ['area_GEO', 'volume_GEO', 'area_volume_ratio_GEO',
			       'edge_ratio_GEO', 'radius_ratio_GEO', 'aspect_ratio_GEO',
			       'max_solid_angle_GEO', 'min_solid_angle_GEO', 'solid_angle_GEO',
			       'area_IBA', 'volume_IBA', 'area_volume_ratio_IBA', 'edge_ratio_IBA',
			       'radius_ratio_IBA', 'aspect_ratio_IBA', 'max_solid_angle_IBA',
			       'min_solid_angle_IBA', 'solid_angle_IBA', 'area_AUC', 'volume_AUC',
			       'area_volume_ratio_AUC', 'edge_ratio_AUC', 'radius_ratio_AUC',
			       'aspect_ratio_AUC', 'max_solid_angle_AUC', 'min_solid_angle_AUC',
			       'solid_angle_AUC']
		else:
			col = ['area_GEO', 'volume_GEO',
			       'area_volume_ratio_GEO', 'edge_ratio_GEO', 'radius_ratio_GEO',
			       'aspect_ratio_GEO', 'max_solid_angle_GEO', 'min_solid_angle_GEO',
			       'solid_angle_GEO', 'area_IBA', 'volume_IBA', 'area_volume_ratio_IBA',
			       'edge_ratio_IBA', 'radius_ratio_IBA', 'aspect_ratio_IBA',
			       'max_solid_angle_IBA', 'min_solid_angle_IBA', 'solid_angle_IBA']
		df_mean = pd.DataFrame()
		df_mean['alphas'] = df1.alphas
		for c in col:
			for i in np.arange(0, df1.shape[0]):
				df_mean.loc[i, c] = (df1.loc[i, c] + df2.loc[i, c] + df3.loc[i, c]) / 3.0
		
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = GEO', fontsize=5)
		ax.set_xlabel('Alpha')
		ax.set_ylabel('Rank')
		t1 = df_mean['alphas']
		t2 = df_mean['alphas']
		t3 = df_mean['alphas']
		t4 = df_mean['alphas']
		t5 = df_mean['alphas']
		t6 = df_mean['alphas']
		t7 = df_mean['alphas']
		t8 = df_mean['alphas']
		t9 = df_mean['alphas']
		
		ft1 = df_mean['area_GEO']
		ft2 = df_mean['volume_GEO']
		ft3 = df_mean['area_volume_ratio_GEO']
		ft4 = df_mean['edge_ratio_GEO']
		ft5 = df_mean['radius_ratio_GEO']
		ft6 = df_mean['aspect_ratio_GEO']
		ft7 = df_mean['max_solid_angle_GEO']
		ft8 = df_mean['min_solid_angle_GEO']
		ft9 = df_mean['solid_angle_GEO']
		
		ax.plot(t1, ft1, color='tab:blue', marker='o', label='area')
		ax.plot(t2, ft2, color='tab:red', marker='o', label='volume')
		ax.plot(t3, ft3, color='tab:green', marker='o', label='area_volume_ratio')
		ax.plot(t4, ft4, color='tab:orange', marker='o', label='edge_ratio')
		ax.plot(t5, ft5, color='tab:olive', marker='o', label='radius_ratio')
		ax.plot(t6, ft6, color='tab:purple', marker='o', label='aspect_ratio')
		ax.plot(t7, ft7, color='tab:brown', marker='o', label='max_solid_angle')
		ax.plot(t8, ft8, color='tab:pink', marker='o', label='min_solid_angle')
		ax.plot(t9, ft9, color='tab:gray', marker='o', label='solid_angle')
		
		leg = ax.legend(loc='upper right')
		leg.get_frame().set_alpha(0.5)
		plt.xticks(range(12))
		plt.savefig(output_dir + kind + '_pic_average_geo.png', dpi=800)
		plt.show()
		plt.close()
		df_mean.to_csv(output_dir + kind + '_pic_average_geo.csv', index=False)
		
		###################
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = IBA', fontsize=5)
		ax.set_xlabel('Alpha')
		ax.set_ylabel('Rank')
		t1 = df_mean['alphas']
		t2 = df_mean['alphas']
		t3 = df_mean['alphas']
		t4 = df_mean['alphas']
		t5 = df_mean['alphas']
		t6 = df_mean['alphas']
		t7 = df_mean['alphas']
		t8 = df_mean['alphas']
		t9 = df_mean['alphas']
		
		ft1 = df_mean['area_IBA']
		ft2 = df_mean['volume_IBA']
		ft3 = df_mean['area_volume_ratio_IBA']
		ft4 = df_mean['edge_ratio_IBA']
		ft5 = df_mean['radius_ratio_IBA']
		ft6 = df_mean['aspect_ratio_IBA']
		ft7 = df_mean['max_solid_angle_IBA']
		ft8 = df_mean['min_solid_angle_IBA']
		ft9 = df_mean['solid_angle_IBA']
		
		ax.plot(t1, ft1, color='tab:blue', marker='o', label='area')
		ax.plot(t2, ft2, color='tab:red', marker='o', label='volume')
		ax.plot(t3, ft3, color='tab:green', marker='o', label='area_volume_ratio')
		ax.plot(t4, ft4, color='tab:orange', marker='o', label='edge_ratio')
		ax.plot(t5, ft5, color='tab:olive', marker='o', label='radius_ratio')
		ax.plot(t6, ft6, color='tab:purple', marker='o', label='aspect_ratio')
		ax.plot(t7, ft7, color='tab:brown', marker='o', label='max_solid_angle')
		ax.plot(t8, ft8, color='tab:pink', marker='o', label='min_solid_angle')
		ax.plot(t9, ft9, color='tab:gray', marker='o', label='solid_angle')
		
		leg = ax.legend(loc='upper right')
		leg.get_frame().set_alpha(0.5)
		plt.xticks(range(12))
		plt.savefig(output_dir + kind + '_pic_average_iba.png', dpi=800)
		plt.show()
		plt.close()
		df_mean.to_csv(output_dir + kind + '_pic_average_iba.csv', index=False)
		
		if kind == 'biclass':
			fig, ax = plt.subplots(figsize=(10, 7))
			ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = AUC', fontsize=5)
			ax.set_xlabel('Alpha')
			ax.set_ylabel('Rank')
			t1 = df_mean['alphas']
			t2 = df_mean['alphas']
			t3 = df_mean['alphas']
			t4 = df_mean['alphas']
			t5 = df_mean['alphas']
			t6 = df_mean['alphas']
			t7 = df_mean['alphas']
			t8 = df_mean['alphas']
			t9 = df_mean['alphas']
			
			ft1 = df_mean['area_AUC']
			ft2 = df_mean['volume_AUC']
			ft3 = df_mean['area_volume_ratio_AUC']
			ft4 = df_mean['edge_ratio_AUC']
			ft5 = df_mean['radius_ratio_AUC']
			ft6 = df_mean['aspect_ratio_AUC']
			ft7 = df_mean['max_solid_angle_AUC']
			ft8 = df_mean['min_solid_angle_AUC']
			ft9 = df_mean['solid_angle_AUC']
			
			ax.plot(t1, ft1, color='tab:blue', marker='o', label='area')
			ax.plot(t2, ft2, color='tab:red', marker='o', label='volume')
			ax.plot(t3, ft3, color='tab:green', marker='o', label='area_volume_ratio')
			ax.plot(t4, ft4, color='tab:orange', marker='o', label='edge_ratio')
			ax.plot(t5, ft5, color='tab:olive', marker='o', label='radius_ratio')
			ax.plot(t6, ft6, color='tab:purple', marker='o', label='aspect_ratio')
			ax.plot(t7, ft7, color='tab:brown', marker='o', label='max_solid_angle')
			ax.plot(t8, ft8, color='tab:pink', marker='o', label='min_solid_angle')
			ax.plot(t9, ft9, color='tab:gray', marker='o', label='solid_angle')
			
			leg = ax.legend(loc='upper right')
			leg.get_frame().set_alpha(0.5)
			plt.xticks(range(12))
			plt.savefig(output_dir + kind + '_pic_average_auc.png', dpi=800)
			plt.show()
			plt.close()
			df_mean.to_csv(output_dir + kind + '_pic_average_auc.csv', index=False)
	
	def run_global_rank(self, filename, kind, release):
		df_best_dto = pd.read_csv(filename)
		df_B1 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline1'].copy()
		df_B2 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline2'].copy()
		df_GEO = df_best_dto[df_best_dto['PREPROC'] == '_Geometric_SMOTE'].copy()
		df_SMOTE = df_best_dto[df_best_dto['PREPROC'] == '_SMOTE'].copy()
		df_SMOTEsvm = df_best_dto[df_best_dto['PREPROC'] == '_smoteSVM'].copy()
		df_original = df_best_dto[df_best_dto['PREPROC'] == '_train'].copy()
		o = 'solid_angle'
		if kind == 'biclass':
			a = 7.0
		else:
			a = 7.5
		
		GEOMETRY = '_delaunay_' + o + '_' + str(a)
		df_dto = df_best_dto[df_best_dto['PREPROC'] == GEOMETRY].copy()
		df = pd.concat([df_B1, df_B2, df_GEO, df_SMOTE, df_SMOTEsvm, df_original, df_dto])
		self.rank_by_algorithm(df, kind, o, str(a), release, smote=True)
		self.rank_dto_by(o + '_' + str(a), kind, release, smote=True)
	
	def overall_rank(self, ext, kind, alpha):
		df1 = pd.read_csv(
				output_dir + 'v1_smote_' + kind + '_results_media_rank_solid_angle_' + str(alpha) + '_' + ext + '.csv')
		df2 = pd.read_csv(
				output_dir + 'v2_smote_' + kind + '_results_media_rank_solid_angle_' + str(alpha) + '_' + ext + '.csv')
		df3 = pd.read_csv(
				output_dir + 'v3_smote_' + kind + '_results_media_rank_solid_angle_' + str(alpha) + '_' + ext + '.csv')
		
		col = ['RANK_ORIGINAL', 'RANK_SMOTE', 'RANK_SMOTE_SVM', 'RANK_BORDERLINE1'
			, 'RANK_BORDERLINE2', 'RANK_GEOMETRIC_SMOTE', 'RANK_DELAUNAY']
		
		df_mean = pd.DataFrame()
		df_mean['ALGORITHM'] = df1.ALGORITHM
		df_mean['unit'] = df1.unit
		for c in col:
			for i in np.arange(0, df1.shape[0]):
				df_mean.loc[i, c] = (df1.loc[i, c] + df2.loc[i, c] + df3.loc[i, c]) / 3.0
		
		df_mean['RANK_ORIGINAL'] = pd.to_numeric(df_mean['RANK_ORIGINAL'], downcast="float").round(2)
		df_mean['RANK_SMOTE'] = pd.to_numeric(df_mean['RANK_SMOTE'], downcast="float").round(2)
		df_mean['RANK_SMOTE_SVM'] = pd.to_numeric(df_mean['RANK_SMOTE_SVM'], downcast="float").round(2)
		df_mean['RANK_BORDERLINE1'] = pd.to_numeric(df_mean['RANK_BORDERLINE1'], downcast="float").round(2)
		df_mean['RANK_BORDERLINE2'] = pd.to_numeric(df_mean['RANK_BORDERLINE2'], downcast="float").round(2)
		df_mean['RANK_GEOMETRIC_SMOTE'] = pd.to_numeric(df_mean['RANK_GEOMETRIC_SMOTE'], downcast="float").round(2)
		df_mean['RANK_DELAUNAY'] = pd.to_numeric(df_mean['RANK_DELAUNAY'], downcast="float").round(2)
		
		df_mean.to_csv(output_dir + 'overall_rank_results_' + kind + '_' + str(alpha) + '_' + ext + '.csv', index=False)
	
	def cd_graphics(self, df, datasetlen, kind):  # TODO
		# grafico CD
		names = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DTO']
		algorithms = regression_list
		
		for i in np.arange(0, len(algorithms)):
			avranks = list(df.loc[i])
			algorithm = avranks[0]
			measure = avranks[1]
			avranks = avranks[2:]
			cd = Orange.evaluation.compute_CD(avranks, datasetlen)
			Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=len(algorithms), textspace=3)
			plt.savefig(output_dir + kind + '_cd_' + algorithm + '_' + measure + '.pdf')
			plt.close()
	
	def read_dir_files(self, dir_name):
		f = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
		return f
	
	def find_best_rank(self, results_dir,release):
		results = self.read_dir_files(results_dir)
		df = pd.DataFrame(columns=[['ARQUIVO', 'WINER']])
		i = 0
		for f in results:
			df_temp = pd.read_csv(results_dir + f)
			df.at[i, 'ARQUIVO'] = f
			best = df_temp.index[df_temp['ALGORITHM'] == 'avarage'].tolist()
			temp = df_temp.iloc[best[0], 1:-1]
			temp = temp.sort_values()
			temp = temp.reset_index()
			df.at[i, 'WINER'] = temp.iloc[0, 0]
			i += 1
		
		df.to_csv(output_dir + release+'_best_ranks.csv',index=False)
	
	def find_best_delaunay(self, results_dir, tipo):
		df = pd.read_csv(results_dir + tipo)
		i = 0
		j = 0
		df_best = pd.DataFrame(columns=['ARQUIVO', 'WINER'])
		win = list(df['WINER'])
		for w in win:
			if w == 'DELAUNAY':
				df_best.at[i, 'ARQUIVO'] = df.iloc[j, 1]
				df_best.at[i, 'WINER'] = df.iloc[j, 2]
				i += 1
			j += 1
		
		df_best.to_csv(output_dir + 'only_best_delaunay_pca_biclass_media_rank.csv')