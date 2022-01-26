import numpy as np
import io
from PIL import Image
import zipfile
import os
import pdb
import pandas as pd
import pickle

class ZipMake:

  def __init__(self, current_dir,zip_path,csv_path,df_filename,dir_name):
    self.current_dir = current_dir

    self.zip_path = zip_path[0]
    self.zip_test_path = zip_path[1]
    
    self.df = pd.read_csv(csv_path[0])
    self.df_test = pd.read_csv(csv_path[1])

    self.df_filename = df_filename[0]
    self.df_test_filename = df_filename[1]

    self.dir_name = dir_name[0]
    self.dir_test_name = dir_name[1]

  #データフレーム作成
  def make_train_data(self,mkfrm_flag = False):

    '''訓練データ用

    引数：mkfrm_flag(default:False)
          mkfrm_flag=Trueとすることによりcsvファイルを読み込んだpadasのデータフレームをpickleとして作成
    
    処理内容：画像を配列化したデータと目的変数の'LOAN_AMOUNT'をタプル型のpickleファイルとして作成
    
    '''
    
    with zipfile.ZipFile(self.zip_path, 'r') as zip_file:
      
      if mkfrm_flag:
        tmp_lst = zip_file.infolist()
        
        #仮の配列格納用のリスト
        train_X = []

        # zipファイル内の各ファイルについてループ
        for tmp_url in tmp_lst:
              # 「zipファイル名/」については処理をしない
              if (tmp_url.filename != tmp_lst[0].filename):                             
                  basename_without_ext = int(os.path.splitext(os.path.basename(tmp_url.filename))[0])
                  df_tmp = self.df[self.df['IMAGE_ID'] == basename_without_ext]
                  train_X.append([int(df_tmp['LOAN_ID']),tmp_url])

        
        df_tmp = pd.DataFrame(train_X,columns=['LOAN_ID','filepath'])

        self.df = pd.merge(self.df, df_tmp)
        
        with open(self.df_filename, 'wb') as f:
          pickle.dump(self.df,f)
    



      # データ読み込み
      with open(self.df_filename,'rb') as f:
          self.df = pickle.load(f)
      
      num = len(self.df)
      k = 100
      df_list = [self.df.loc[i:i+k-1, :] for i in range(0,num,k)]

      # 同じ数でfile名を作成
      pickle_path2 = [] 
      for i in range(0,len(df_list)):
        pickle_path2.append(self.dir_name+ '/' + os.path.split(self.dir_name)[1]+str(i)+'.pickle')
            
      # 書き込みモードでパスを開いて画像データを作成
      for path, df in zip(pickle_path2, df_list):
        
        with open(path, 'wb') as f:
            img_tmp_array = []        
            loan_tmp_array = np.array([loan for loan in df['LOAN_AMOUNT']])        
            for info in df['filepath']:
                        
                # 対象の画像ファイルを開く
                with zip_file.open(info.filename) as img_file:
                    # 画像のバイナリデータを読み込む
                    img_bin = io.BytesIO(img_file.read())
                    # バイナリデータをpillowから開く
                    img = Image.open(img_bin)
                    # 画像データを配列化
                    img_array = np.array(img)
                    # 格納用のListに追加
                    img_tmp_array.append(img_array)
                        
            data_tuple = tuple((loan_tmp_array,np.array(img_tmp_array)))
            pickle.dump(data_tuple,f)


  # テストデータ作成
  def make_test_data(self,mkfrm_flag=False):

    '''テストデータ用
    引数：mkfrm_flag(default:False)
          mkfrm_flag=Trueとすることによりcsvファイルを読み込んだpadasのデータフレームをpickleとして作成
    
    処理内容：画像を配列化したデータをpickleファイルとして作成
    
    '''

    with zipfile.ZipFile(self.zip_test_path, 'r') as zip_file:
      

      if mkfrm_flag:
        tmp_lst = zip_file.infolist()
        
        #仮の配列格納用のリスト
        train_X = []

        # zipファイル内の各ファイルについてループ
        for tmp_url in tmp_lst:
              # 「zipファイル名/」については処理をしない
              if (tmp_url.filename != tmp_lst[0].filename):                
                  basename_without_ext = int(os.path.splitext(os.path.basename(tmp_url.filename))[0])
                  df_tmp = self.df_test[self.df_test['IMAGE_ID'] == basename_without_ext]
                  train_X.append([int(df_tmp['LOAN_ID']),tmp_url])

        
        df_test = pd.DataFrame(train_X,columns=['LOAN_ID','filepath'])
        self.df_test = pd.merge(self.df_test,df_test)

        with open(self.df_test_filename, 'wb') as f:
          pickle.dump(self.df_test,f)
    
      # データ読み込み
      with open(self.df_test_filename,'rb') as f:
          self.df_test = pickle.load(f)
      
      num = len(self.df_test)
      k = 100
      df_list = [self.df_test.loc[i:i+k-1, :] for i in range(0,num,k)]

       #指定した長さでデータpathを作成
      pickle_path2 = [] 
      for i in range(0,len(df_list)):
        pickle_path2.append(self.dir_test_name+ '/' + os.path.split(self.dir_test_name)[1]+str(i)+'.pickle')
            

      for path, df in zip(pickle_path2, df_list):
        
        with open(path, 'wb') as f:
            img_tmp_array = [] 
            for info in df['filepath']:
                         
                # 対象の画像ファイルを開く
                with zip_file.open(info.filename) as img_file:
                    # 画像のバイナリデータを読み込む
                    img_bin = io.BytesIO(img_file.read())
                    # バイナリデータをpillowから開く
                    img = Image.open(img_bin)
                    # 画像データを配列化
                    img_array = np.array(img)
                    # 格納用のListに追加
                    img_tmp_array.append(img_array)

            pickle.dump(np.array(img_tmp_array),f)

    



      