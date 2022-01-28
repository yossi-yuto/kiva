# kiva
csv形式のデータと画像に対してマルチモーダルなデータセットの作成方法と前処理

## 使用方法
前と同様にmain.pyとZipMake.pyをダウンロード//
main.pyを実行（ZipMake.py）は同じディレクトリ内に保存

### 以前との変更点
- 各自の環境下でファイルパスの指定が不要
- 王さんの前処理を反映

### 手順
1. 訓練・テスト用の画像zipファイルとcsvファイルの合計４つを同じディレクトリ内に置く
2. isMakeDataフラグをTrueにするとデータ作成、作成後はFalseのまま
3. ZipMakeのメソッドのmake_train_data,make_test_dataはTrueに設定変更