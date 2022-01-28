# kiva
画像データの作成と前処理

## 使用
main.pyとZipMake.pyをダウンロード
main.pyを実行（ZipMake.py）は同じディレクトリ内に保存

### 以前との変更点
- 各自の環境下でファイルパスの指定が不要
- 訓練・テスト用の画像zipファイルとcsvファイルの合計４つを同じディレクトリ内に置く

### 手順
1. isMakeDataフラグをTrueにするとデータ作成、作成後はFalseのまま
2. ZipMakeのメソッドのmake_train_data,make_test_dataはTrueに設定変更
