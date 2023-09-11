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

### model/ccae.py

Continuous Convolutional Auto Encoder (CCAE) のモデル構造を記述。

https://arxiv.org/abs/2101.06742

### model/ae.py

シンプルな Auto Encoder モデルを記述。

### utils/data_preprocessor.py

クラス `DataPreprocessor` に
- CSVデータの読み込み(1回目)
- キャッシュデータの読み込み(2回目以降)
- **モデルで学習するため**のデータの前処理(データ自体の大きな加工処理、および加工したデータの保存などは`DataPreprocessor`ではなく``で行う)

などの機能をまとめている。

### config
