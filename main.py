from bdb import set_trace
from calendar import firstweekday
import csv
import pickle
import glob
import os 
import pdb
import matplotlib.pyplot as plt
from ZipMake import ZipMake
from ZipMake import OpenPickle

import pandas as pd
import random
import time
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

isMakeData = True  # データを作成するときはTrue、データ作成後はFalse

#============================================================
# tensorflow2.xでのGPUの設定
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    #
    for k in range(len(physical_devices)):
        tf.config.set_visible_devices(physical_devices[k], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
else:
    print("Not enough GPU hardware devices available")
#============================================================
#=================================================================

#=================================================================
# パスの設定
# 各自の環境によって適切に設定

# current directory path
current_dir = os.getcwd()  # ファイルを作成する場所（path）の指定

'''
# フォルダ内のpickleファイルをすべて削除
# 削除するフォルダのパス
remove_path = os.path.join(current_dir,'*.pickle')
for file in glob.glob(remove_path):
  os.remove(file)
'''

# zip path
zip_train_path = os.path.join(current_dir, "train_images.zip") #訓練データのzipパス
zip_test_path = os.path.join(current_dir, "test_images.zip")#テストデータのzipパス

# csv path
csv_train_path  = os.path.join(current_dir, "train.csv") # 訓練データのcsvパス
csv_test_path = os.path.join(current_dir, "test.csv")  # テストデータのcsvパス

# csv to pickle 
df_train_filename = os.path.join(current_dir,"dataframe_train.pickle") #  訓練csvからpickleに変えるファイルパス
df_test_filename = os.path.join(current_dir,"dataframe_test.pickle")   #  テストcsvからpickleに変えるファイルパス

# 画像をpickleデータとして展開先のpath(新規作成したフォルダをそれぞれ指定)
dir_train_name = os.path.join(current_dir,"train")  # 画像データ（訓練用）のpickleの展開先のパス
dir_test_name = os.path.join(current_dir,"test")   # 画像データ（テスト用）のpickleの展開先のパス 

if not os.path.exists(dir_train_name):
  os.makedirs(dir_train_name)
if not os.path.exists(dir_test_name):
  os.makedirs(dir_test_name)

#パスのリスト化
#=========================================================================
zip_path = list([zip_train_path,zip_test_path])
csv_path = list([csv_train_path,csv_test_path])
df_filename = list([df_train_filename,df_test_filename])
dir_name = list([dir_train_name,dir_test_name])
#=========================================================================

#データ作成
#==========================================================================
if isMakeData:
  # zipmakeインスタンスを作成
  zipmake = ZipMake.ZipMake(current_dir,zip_path,csv_path,df_filename,dir_name)

  # 訓練データを作成（mkfrm_flgはdataframe.pikcleを作成するときTrue,default:False)
  zipmake.make_train_data(mkfrm_flag=True)

  # テストデータを作成（mkfrm_flgはdataframe.pikcleを作成するときTrue,default:False)
  zipmake.make_test_data(mkfrm_flag=True)
#===========================================================================



# テーブルデータの前処理
#====================================================================
'''
with open(df_filename[0],'rb') as train, open(df_filename[1],'rb') as test:
  df_train = pickle.load(train) # 訓練データのデータフレーム
  df_test = pickle.load(test)   # テストデータのデータフレーム
'''
df_train = pd.read_csv(csv_path[0])
df_test = pd.read_csv(csv_path[1])

#欠損値を消す(TOWN_NAMEとTAGS）
#TOWN_NAMEを取り入れるより、Countrycodeで判別できると判断したため、削除
df_train.drop('TOWN_NAME', axis=1, inplace=True)
df_test.drop('TOWN_NAME', axis=1, inplace=True)

df_train.drop('TAGS', axis=1, inplace=True)
df_test.drop('TAGS', axis=1, inplace=True)

#欠損値を埋める（72195行に欠損値があるため、言語は英語なので、DESCRIPTIONと同じ内容に代入）
df_train['DESCRIPTION_TRANSLATED'].fillna(df_train['DESCRIPTION'], inplace=True)
df_test['DESCRIPTION_TRANSLATED'].fillna(df_test['DESCRIPTION'], inplace=True) 

#欠損値を埋める（CURRENCY_POLICYがスタンダードの場合は全部NAとなっているため、CURRENCYは変換しなくていいという認識で０で埋める）
df_train['CURRENCY_EXCHANGE_COVERAGE_RATE'].fillna(0)
df_test['CURRENCY_EXCHANGE_COVERAGE_RATE'].fillna(0)

#CountrycodeとCOUNTRY_NAMEは同じ意味なのでCOUNTRY_NAME削除
df_train.drop('COUNTRY_NAME', axis=1, inplace=True)
df_test.drop('COUNTRY_NAME', axis=1, inplace=True)

#SECTOR_NAMEと似たような表現なので削除
df_train.drop('ACTIVITY_NAME', axis=1, inplace=True)
df_test.drop('ACTIVITY_NAME', axis=1, inplace=True)

#文字表記を数字に変換
#言語
df_train['ORIGINAL_LANGUAGE'] = df_train['ORIGINAL_LANGUAGE'].replace({'English':1, 'Spanish':2, 'French':3, 'Portuguese':4, 'Russian':5})
#使用用途
df_train['SECTOR_NAME'] = df_train['SECTOR_NAME'].replace({'Agriculture':1, 'Food':2, 'Retail':3, 'Housing':4, 'Services':5,
                                               'Clothing':6, 'Personal Use':7, 'Education':8, 'Arts':9, 'Health':10,
                                               'Transportation':11, 'Construction':12, 'Manufacturing':13, 'Entertainment':14, 'Wholesale':15})
#通貨方針
df_train['CURRENCY_POLICY'] = df_train['CURRENCY_POLICY'].replace({'shared':1, 'standard':0})
#返済間隔
df_train['REPAYMENT_INTERVAL'] = df_train['REPAYMENT_INTERVAL'].replace({'monthly':1, 'bullet':2, 'irregular':3})
#融資形態
df_train['DISTRIBUTION_MODEL'] = df_train['DISTRIBUTION_MODEL'].replace({'field_partner':1, 'direct':2})
#===========================================================================================================



# 訓練データの生成-----------------------------
# 画像の前処理
# 画像：tf.data.Dataset / テーブルデータ：df_train
#================================================================
# 画像データのpickleパスを訓練、テストのフォルダから全部取得
pickle_file = glob.glob(os.path.join(dir_name[0], '*.pickle'))     
pickle_test_file = glob.glob(os.path.join(dir_name[1],'*.pickle'))
open = OpenPickle(pickle_file)
id , image_and_label = open.open_pkl_image()
df_train = pd.merge(df_train,id,on='LOAN_ID')
#================================================================

pdb.set_trace()

# モデル学習
#==================================================
'''
・画像データのモデル
   model.fit(image_and_label) 画像と正解値のセットなので分離は不要

・テーブルデータのモデル
   model.fit(df_train)  データフレームを訓練データ

・マルチモーダルなモデル
   ?.fit(image_and_label,df_train)  マルチモーダルなモデルについては詳しくないのでよろしくお願いいたします。

などのモデルはここに記述↓

'''
#===================================================



# テストデータの作成とモデル予測--------------------------------------
#================================================================
for path in pickle_test_file:
  id, image_test = open.open_test_pkl_image(path)
  df_test = pd.merge(df_test,id,on='LOAN_ID')

  # 学習モデルに適用し、予測する
  # 1.画像用モデルには model.predict(image_test)
  # 2.テーブルデータ用のモデルには model.predict(df_test)
  # (3).マルチモーダルなモデルを使用する場合は、そのモデルに対して  model.predict(image_test,df_test) 

  # idにはLOAN_IDが格納されているのでsample_submission.csvにはfor分で逐次的に予測したデータを書き込んでいく。
  '''
  submit_path = os.path.join(current_dir, "sample_submission.csv")
  submit = pd.read_csv(submit_path)
  submit = pd.merge(id,summit,on='LOAN_ID')
  submit['LOAN_AMOUNT'] = pred  ←予測したデータ
  submit.to_csv(submit_path)  ←提出用データの上書き保存 
  
  '''
#====================================================================
  

pdb.set_trace()

