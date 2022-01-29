# kiva クラウドファンディング資金予測
csv形式のデータと画像に対してマルチモーダルなデータセットの作成方法と前処理

## データ概要
画像データのあるテーブルデータは訓練、テストデータともに99%以上存在

### 訓練データの概要
- 画像ファイルが存在するデータ数：91029
- テーブルデータの総数：91333
### テストデータの概要
- 画像ファイルが存在するデータ数：91819
- テーブルデータの総数：91822

## 使用方法
前と同様にmain.pyとZipMake.pyをダウンロードし、main.pyを実行

ただしZipMake.pyは同じディレクトリ内に保存


### 以前との変更点
- 各自の環境下でファイルパスの指定が不要
- ZipMake.pyの内容を変更（LOAN_IDで画像データとテーブルデータを参照）
- テーブルデータについての読み込みと前処理（王さんの前処理）を反映
- tensorflowのtf.data.Datasetを利用しているのでmodel.fitの引数は一つ

### 手順
1. 訓練・テスト用の画像zipファイルとcsvファイルの合計４つを同じディレクトリ内に置く
2. isMakeDataフラグをTrueにするとデータ作成、作成後はFalseのまま
3. ZipMakeのメソッドのmake_train_data,make_test_dataはTrueに設定変更
