'''

## imports
import pandas as pd


# 16,185 samples

# train: 12,948 (current: 8144)
# test: 3,237   (current: 8041)
#     --> so move 4,804 from test to train


## generate dataframes
df_train = pd.read_csv('train.csv').drop('Unnamed: 0', axis=1) ## drop old index column
df_test = pd.read_csv('test_labels.csv').drop('Unnamed: 0', axis=1)

## get last 4,804 samples from test (to be integrated into our training set)
to_move = df_test.tail(4804)
# print(to_move.head())

## get first 3,237 samples from test (our new test set)
df_test = df_test.head(3237)

## add new col to hold buffer (so no name conflicts. new name = index + len(train))
## note: these are all the photos that we need to remove
buffer = pd.DataFrame(to_move.index+1, columns={'img'})  ## dataframe holding new file name numbers
buffer['fname'] = buffer.index + len(df_train) + 1       ## let start of added file names be after the last train sample
buffer['fname'] = buffer['fname'].apply(lambda x: ('0' + str(x) + '.jpg') if x < 9999 else (str(x) + '.jpg')) ## add correct full file name

# print(buffer.head())
# print(buffer.tail())

## to iterate through the images in test and rename them for their approproate train names

## imports
import os
import shutil

## get directory path (cars_test folder, below the current directory)
directory = os.getcwd() + '/cars_test/'


## set all image names to numbers (08144, 12948)
x = -1
## only iterate through all images to be moved
# for i in range(3238, 8042):
#     ## x = offset from end of test set
#     x += 1
#     ## conditional to make leading 0s accurate
#     if (x + len(df_train) > 9999):
#         val = str(x + len(df_train))
#     else:
#         val = '0' + str(x + len(df_train))
#     ## rename the files
#     old_name = directory + '0' + str(i) + '.jpg'
#     new_name = directory + val + '.jpg'
#     os.rename(old_name, new_name)

# for i in range(8145, 12949):

#     if i > 9999:
#         val = str(i) + '.jpg'
#         old_loc = directory + val

#     else:
#         val = '0' + str(i) + '.jpg'
#         old_loc = directory + val

#     new_loc = os.getcwd() + '/cars_train/'
#     new_loc += val
    
#     shutil.move(old_loc, new_loc)


##  now bring the df's back together 

## new test df
df_new_test = df_test

## new train df
to_move.reset_index(inplace = True, drop = True)    ## reset index to start from 0 for proper merge
to_move['fname'] = buffer['fname']  ## merge name cols
df_new_train = pd.concat([df_train, to_move])   ## concatenate the old and new training samples

## write to csv files
df_new_test.to_csv('new_test.csv')
df_new_train.to_csv('new_train.csv')

'''