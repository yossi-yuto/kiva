import pickle
import glob
import os 
import pdb
import matplotlib.pyplot as plt
import ZipMake
import pandas as pd
import random
import time

isMakeData = True    # データを作成するときはTrue、データ作成後はFalse

#=================================================================
# パスの設定
# 各自の環境によって適切に設定

# current directory path
current_dir = "C:/Users/yu-toyosi/kiva"   # ファイルを作成する場所（path）の指定

'''
# フォルダ内のpickleファイルをすべて削除
# 削除するフォルダのパス
remove_path = os.path.join(current_dir,'*.pickle')
for file in glob.glob(remove_path):
  os.remove(file)
'''

# zip path
zip_train_path = "C:/Users/yu-toyosi/kiva/train_images.zip" #訓練データのzipパス
zip_test_path = "C:/Users/yu-toyosi/kiva/test_images.zip" #テストデータのzipパス

# csv path
csv_train_path  = "C:/Users/yu-toyosi/kiva/train.csv" # 訓練データのcsvパス
csv_test_path = "C:/Users/yu-toyosi/kiva/test.csv"  # テストデータのcsvパス

# csv to pickle 
df_train_filename = "C:/Users/yu-toyosi/kiva/dataframe_train.pickle" #  訓練csvからpickleに変えるファイルパス
df_test_filename = "C:/Users/yu-toyosi/kiva/dataframe_test.pickle"   #  テストcsvからpickleに変えるファイルパス

# 画像をpickleデータとして展開先のpath(新規作成したフォルダをそれぞれ指定)
dir_train_name = "C:/Users/yu-toyosi/kiva/train"  # 画像データ（訓練用）のpickleの展開先のパス
dir_test_name = "C:/Users/yu-toyosi/kiva/test"    # 画像データ（テスト用）のpickleの展開先のパス 

#=========================================================================
zip_path = list([zip_train_path,zip_test_path])
csv_path = list([csv_train_path,csv_test_path])
df_filename = list([df_train_filename,df_test_filename])
dir_name = list([dir_train_name,dir_test_name])

start = time.time() # 時間計測

if isMakeData:
  # zipmakeインスタンスを作成
  zipmake = ZipMake.ZipMake(current_dir,zip_path,csv_path,df_filename,dir_name)

  # 訓練データを作成（mkfrm_flgはdataframe.pikcleを作成するときTrue,default:False)
  zipmake.make_train_data(mkfrm_flag=True)

  # テストデータを作成（mkfrm_flgはdataframe.pikcleを作成するときTrue,default:False)
  zipmake.make_test_data(mkfrm_flag=True)



with open(df_filename[0],'rb') as train, open(df_filename[1],'rb') as test:
  df_train = pickle.load(train) # 訓練データのデータフレーム
  df_test = pickle.load(test)   # テストデータのデータフレーム


# 画像データのpickleパスを訓練、テストのフォルダから全部取得
pickle_file = glob.glob(os.path.join(dir_name[0], '*.pickle'))     
pickle_test_file = glob.glob(os.path.join(dir_name[1],'*.pickle'))
with open(random.choice(pickle_file), 'rb') as f, open(pickle_test_file, 'rb') as t: # pickleファイルからランダムに1つを抽出
  # ファイルの読み込み
  x = pickle.load(f)      
  y = pickle.load(t)      


# 100枚の訓練画像データの配列 (100, 224, 224, 3)     
train_x = x[1] / 255  # 正規化済
train_y = x[0]  # 正解ラベル (100,)
# 100枚のテスト用画像データの配列
test_x = y / 255   #  正規化済 (100, 224, 224, 3)

elapsed_time = time.time() - start
print( "elaspsed_time:{0}".format(elapsed_time) + "[sec]")

pdb.set_trace()


