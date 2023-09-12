# Feature-Extraction
Repository for Auto Encoder model to extract tactile feature

## File Structure
- `Feature-Extraction/`ディレクトリと`dataset/`ディレクトリを同一階層のワークスペースに配置すること。

## Quick Start
```
python train_ccae.py -y ./config/ccae.yaml
```

## Description

### train_ccae.py

### trainer.py

### model/
#### ccae.py

Continuous Convolutional Auto Encoder (CCAE) のモデル構造を記述。

https://arxiv.org/abs/2101.06742

#### ae.py

シンプルな Auto Encoder モデル構造を記述。

### utils/
#### data_preprocessor.py

クラス `DataPreprocessor` に
- CSVデータの読み込み(1回目)
- キャッシュデータの読み込み(2回目以降)
- 正規化・標準化などのスケーリング処理


などの**モデルで学習する直前のデータの前処理機能**をまとめている。

なお、データの平滑化処理・外れ値処理・順運動学による3次元座標の計算・パッチの重心座標の計算といったデータセットの大規模な加工・拡張・新しいCSVデータセットとしての再保存については`DataPreprocessor`ではなく``で行う。

### config/

#### ae.yaml

#### ccae.yaml

### weight/

学習したモデルの重みパラメータは `weight/` ディレクトリに保存される。

## Tips
### 処理速度の高速化
- csv形式で保存されているデータセットを扱うためにPandasを用いるが、**NumpyやPytorchと比較して処理速度がとても遅いため注意**。基本的にデータセットを読み込んだあとは早めにPytorchの`tensor`型に変換した方が良い。（将来的にはROSのプログラムを整備して、PickleなどのPandas以外のバイナリファイルでデータ収集を行うべき）

参考： https://propen.dream-target.jp/blog/pandas

- Pytorch2.0以降の機能`torch.compile`を用いることで、モデルの高速化ができる。

参考： https://www.mattari-benkyo-note.com/2023/03/18/torch-compile/
