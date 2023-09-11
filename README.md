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

シンプルな Auto Encoder モデルを記述。

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