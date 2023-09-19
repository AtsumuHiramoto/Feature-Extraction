# Feature-Extraction
Repository for Auto Encoder model to extract tactile feature

## File Structure
- `Feature-Extraction/`ディレクトリと`dataset/`ディレクトリを同一階層のワークスペースに配置すること。

- datasetはWasedaBoxに保存されている。
https://waseda.box.com/s/dhzvtj80np5xtazcrlaaof0fn2kqcoyv

## Quick Start
Example(Ubuntu):
```
mkdir workspace # make workspace
cd workspace
git clone https://github.com/AtsumuHiramoto/Feature-Extraction.git # clone from github
mkdir dataset # make dataset directory on workspace 
```
Then, download dataset from WasedaBox.
https://waseda.box.com/s/dhzvtj80np5xtazcrlaaof0fn2kqcoyv

After that, move dataset on `workspace/dataset/`

### Train/Test Basic LSTM model
```
cd workspace/Feature-Extraction
```
Training LSTM
```
python main.py -y ./config/lstm.yaml -m "Train"
```
Testing LSTM (Joint Prediction)
```
python main.py -y ./config/lstm.yaml -m "Test"
```

### Train Continuous Convolutional Auto Encoder
Under construction
```
python train_ccae.py -y ./config/ccae.yaml
```

## Description

### main.py
メインプログラム。モデルの学習、テストはこのプログラムを起動して行う。

### trainer.py

### bptt_trainer.py
- 誤差逆伝播によるモデルの学習
- 実際の関節角度と予測関節角度の比較
- スケーリングされたデータの復元（Rescaling）

### layer/
モデル構造を記述

#### ccae.py

Continuous Convolutional Auto Encoder (CCAE) のモデル構造を記述。

参考: https://arxiv.org/abs/2101.06742

#### ae.py

シンプルな Auto Encoder モデル構造を記述。

### utils/
詳細は`utils/README.md`を参照。

#### data_preprocessor.py

クラス`DataPreprocessor`に
- CSVデータの読み込み・読み込んだCSVデータのキャッシュデータを作成(1回目)
- キャッシュデータの読み込み(2回目以降)
- 正規化・標準化などのスケーリング処理

などの**モデルで学習する直前のデータの前処理機能**をまとめている。

なお、データの平滑化処理・外れ値処理・順運動学による3次元座標の計算・パッチの重心座標の計算といったデータセットの大規模な加工・拡張・新しいCSVデータセットとしての再保存については`DataPreprocessor`ではなく`HandlingDataMaker`で行う。

#### handling_data_maker.py
Under construction

クラス`HandlingDataMaker`に
- データの平滑化処理
- データの外れ値処理
- 加工・拡張したデータを新しいCSVデータセットとして保存

などの**CSVデータセット自体の加工機能**をまとめている。

#### tactile_coordinates_manager.py
Under construction

クラス`TactileCoordinatesManager`に
- 順運動学の計算による3次元座標の計算
- パッチの重心座標の計算

といった機能をまとめている。

#### callback.py
eiplからForkしている

- Early stopping
の機能をまとめている。

#### make_dataset.py

- DataLoaderで読み込まれるデータセットの作成
- 入力関節にガウシアンノイズを付与

#### visualizer.py

- training, test lossの可視化、保存

### config/

#### lstm.yaml
Basic LSTMの学習、テストを行う際のConfigファイル。

#### ae.yaml

#### ccae.yaml

### weight/
学習したモデルの重みパラメータ・学習時のスケーリングパラメータ・予測関節角度のグラフは `weight/` ディレクトリに保存される。

## Tips
### 処理速度の高速化
深層学習モデルの学習には時間がかかるため、可能な限り処理速度を上げることが大事である。1週間かかっていたモデルの学習でも、コードの工夫しだいで1日や数時間で学習できるようになることもある。

- まずはこの記事に目を通すこと。深層学習モデルを学習する上での基本的な高速化手法が書かれている。

参考: https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587

- csv形式で保存されているデータセットを読み込むためにPandasを用いるが、**NumpyやPytorchと比較して処理速度がとても遅いため注意**。基本的にデータセットを読み込んだあとは早めにPytorchの`tensor`型などに変換した方が良い。（将来的にはROSのプログラムを整備して、PickleなどのPandas以外のバイナリファイルでデータ収集を行うべき）

参考： https://propen.dream-target.jp/blog/pandas

- Pytorch2.0以降の機能`torch.compile`を用いることで、モデルの高速化ができる。

参考： https://www.mattari-benkyo-note.com/2023/03/18/torch-compile/

### プログラミングのコツ

- 読めるコードを書くこと。

参考: https://qiita.com/KNR109/items/3b14e2e8f89a33c0f959

- PEP8を守ること。

参考： https://qiita.com/simonritchie/items/bb06a7521ae6560738a7

- 命名規則を守ること。

参考： https://qiita.com/naomi7325/items/4eb1d2a40277361e898b

- Gitでコードを管理すること。

参考: https://qiita.com/jesus_isao/items/63557eba36819faa4ad9

TODO

- データセットの整形、Dataloaderを使うために→完了
- Lossの適用範囲にマスクをかけて後半のコピー部分を含めなくする
- 予測Timestepを何Timestep先にするかのパラメータ作成
- データのHzを調整