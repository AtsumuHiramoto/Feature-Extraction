# utils
データ処理の機能を持つプログラムを管理するディレクトリ

## data_pre_processor.py
クラス`DataPreprocessor`に**モデルで学習する直前のデータの前処理機能**をまとめている。

**データセットの加工・拡張・保存などの機能**については`DataPreprocessor`ではなく`HandlingDataMaker`にまとめている。

###  `def __init__(self, load_dir, input_data)`
- `DataPreprocessor`クラスのコンストラクタ。
- ここで今回読み込みたいデータセットのパスや、モデルの学習に用いる入力データの種類を指定する。

### `def load_handling_dataset(self)`
- データを読み込む関数。
- プログラム実行の1回目は、指定した`load_dir`内部のCSVデータを読み込んで`self.handling_data`を作成する。また、作成された`self.handling_data`のキャッシュデータ`data_cache.pkl`を保存するため、プログラムを2回目以降に実行する場合はCSVデータの代わりにキャッシュデータを読み込むことでスムーズに`self.handling_data`を取得できる。
- キャッシュデータとともに、`self.handling_data`を構成するCSVファイルの情報を保持した`data_cache_info.json`を生成する。このファイルの情報を検証することで、キャッシュデータを読み込んでも大丈夫かどうか（以前作成した`self.handling_data`と今回取得しようとしている`self.handling_data`が同一であるか）を判定している。

### `def scaling_handling_dataset(scaling_mode, scaling_range, separate_axis, separate_joint)`
- データセットのスケーリングを行う関数。
- 関数内部では、CSVデータセットのヘッダーと正規表現でのマッチングによってスケーリングの範囲を指定しているため、データセットのヘッダー名を変更した場合は注意が必要。

例：以下のプログラムでは正規表現によるマッチングでスケーリングする触覚データの範囲を指定している。
```
hand_3d_column = [bool(re.match(".*Tactile.*", s)) for s in self.handling_data["columns"]]
```

### `def split_handling_dataset`