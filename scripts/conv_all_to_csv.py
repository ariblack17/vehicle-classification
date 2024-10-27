'''

## imports
from scipy.io import loadmat
import pandas as pd

## load the .mat files
mat_meta = loadmat('cars_meta.mat', simplify_cells=True)['class_names'] ## get all class names
mat_train = loadmat('cars_train_annos.mat', simplify_cells=True)['annotations']
mat_test = loadmat('cars_test_annos.mat', simplify_cells=True)['annotations']
mat_test_labels = loadmat('cars_test_annos_withlabels.mat', simplify_cells=True)['annotations']

## extract all data
df_meta = pd.DataFrame(mat_meta, columns={'class_name'})
df_meta.index += 1

df_train = pd.DataFrame(mat_train) 
df_test = pd.DataFrame(mat_test)
df_test_labels = pd.DataFrame(mat_test_labels)
# note: classes are off by 1! index is 0

## reformat the dataframe
df_train.columns = ['x1', 'y1', 'x2', 'y2', 'class', 'fname']
df_test.columns = ['x1', 'y1', 'x2', 'y2', 'fname']
df_test_labels.columns = ['x1', 'y1', 'x2', 'y2', 'class', 'fname']

## show the dataframes
# print(df_train.head())
# print(df_test.head())

## write to csv
df_meta.to_csv('meta.csv')
df_train.to_csv('train.csv')
df_test.to_csv('test.csv')
df_test_labels.to_csv('test_labels.csv')

'''