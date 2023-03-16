# 陰関数表現を用いた構造化モデル生成法

このレポジトリは修士論文（題目：「陰関数表現を用いた同一構造を持つ3次元物体メッシュモデル生成法の構築」）の
提案手法を実装したものです．


## 引用

引用する際は以下のbibtexを参考にしてください．
```
@article{itayacvim2022,
  title = {陰関数表現を用いた同一構造を持つ3次元物体メッシュモデル生成法の構築},
  author = {響, 板谷 and 翔子, 宮内 and 健一, 諸岡},
  journal = {情報処理学会 コンピュータビジョンとイメージメディア研究会 (CVIM2022)},
  month = {11},
  year = {2022}
}
```

## Requirements
* Ubuntu 20.04 
* Pytorch (tested on 1.10.0)
* plyfile
* matplotlib
* ninja
* pathos
* tensorboardX
* point_cloud_utils


## 環境構築
[conda](https://conda.io/en/latest/)コマンドを用いて仮想環境をインストール：
```bash
conda env create -f environment.yml
conda activate deepsdf
```

## Demo
論文内で使用する4クラス（bathtub，bottle，car，sofa）についての学習済み重みは`examples`以下にあります．
このレポジトリでは，実験用ディレクトリとしてクラスごとに`examples/{classnames}_dit（学習時の重み・ログ保存など）`
を使用します．

以降では，例としてsofaに対するコマンド例を示します．
また，以降ではsofaを例に挙げています．

入力点群に対する構造化モデル生成の実行例：
```bash
#** レポジトリ内共通変数 **
GPU_ID=0
preprocessed_data_dir=/mnt/nas/3DModelDataset/ShapeNet  # 絶対パスの使用を推奨
#** レポジトリ内共通変数 **

modelId=example_sofa
python predict_sdf.py -d ./data -f ${modelId}.ply
CUDA_VISIBLE_DEVICES=${GPU_ID} python reconstruct_structured_meshes.py -d ./data -c latest -m ${modelId} --batch_split 2
```
`オプションbatch_splitの値はGPUのメモリ容量に応じて変更する必要がありますが，
小さい値の方がプログラム中のループ回数が減少し高速に構造化モデルを生成可能です．`

上記の実行により，出力用ディレクトリ`./data/MeshesWithPoints`に入力点群に対する構造化モデルが`${modelId}.ply`ファイルとして生成されます．

## データ作成・前処理
3次元CADモデルの公開データセットである[ShapeNet](https://www.shapenet.org)を使用します．
ShapeNetCore.v2をダウンロードし，`3DModelDataset/ShapeNet`以下に置いています．

また，データセットとして使用するモデルIDは`examples/splits`以下に`.json`形式で格納します．

### CADモデルの符号付き距離場
CADモデルから計算した符号付き距離場のデータを`3DModelDataset/ShapeNet/SdfSamples`以下に，
また，各CADモデルの単位球への正規化パラメータを`3DModelDataset/ShapeNet/NormalizationParameters`以下に
それぞれ`.npz`ファイルとして配置しています．

これらの符号付き距離場，正規化パラメータの計算には[DeepSDF](https://github.com/facebookresearch/DeepSDF)を使用します．
DeepSDFの環境構築方法については`DeepSDF_環境構築.txt`を参照してください．

実行例：
```bash
# CADモデルの符号付き距離場を生成
python preprocess_data.py --data_dir ${preprocessed_data_dir} --source ${preprocessed_data_dir}/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_sofas_train.json --skip
# CADモデルの表面から点群をサンプリング
python preprocess_data.py --data_dir ${preprocessed_data_dir} --source ${preprocessed_data_dir}/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_sofas_train.json --skip --surface
```

### 点群の符号付き距離場
入力点群から計算した符号付き距離場のデータを`3DModelDataset/ShapeNet/PointCloudSdfSamples`以下に置いています．
符号付き距離場の計算には`make_sdf_dataset.py`を使用します：
```bash
python make_sdf_dataset.py -d ${preprocessed_data_dir} -s examples/splits/sv2_sofas_test.json
```
オプション：
* `--missing`：欠損を含む点群を扱う場合
* `--noise`：ノイズを含む点群を扱う場合


## 学習

学習データは，DeepSDFを用いてCADモデルから計算した符号付き距離場です．
手法の学習部分は以下の3段階で構成されます．
1. [Deep Implicit Templates](https://github.com/ZhengZerong/DeepImplicitTemplates)の学習
1. 形状変形ネットワークの学習
1. テンプレート形状に対する構造化モデルの生成

### 1. Deep Implicit Templatesの学習
[Deep Implicit Templates](https://github.com/ZhengZerong/DeepImplicitTemplates)を学習し，
テンプレート空間を求めます．
ネットワークパラメータ・学習パラメータは`examples/sofas/specs.json`を参照してください．
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_deep_implicit_templates.py -e examples/sofas_dit --debug --batch_split 2 -c latest -d ${preprocessed_data_dir}
```

#### Deep Implicit Templatesの学習結果の確認
```bash
# テンプレート形状の確認
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_template_mesh.py -e examples/sofas_dit --debug 
# 各モデルの形状復元結果の確認（トポロジー・頂点数はモデルごとに異なる）
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_training_meshes.py -e examples/sofas_dit --debug --start_id 0 --end_id 20 --octree --keep_normalization
# 各モデルのテンプレート空間における写像結果の確認
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_canonical_positions.py -e examples/sofas_dit --debug --start_id 0 --end_id 20
# 全モデル間のテンプレート空間での対応関係の確認（全モデルをテンプレート空間の座標値をもとに色付けする）
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_meshes_correspondence.py -e examples/sofas_dit --debug --start_id 0 --end_id 20
```

### 2. 形状変形ネットワーク（逆ワーピング関数）の学習
テンプレート空間から入力空間への逆写像を行う逆ワーピング関数を学習します：
ネットワークパラメータ・学習パラメータは`examples/sofas/specs.json`を参照してください．
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_inversed_function.py -e examples/sofas_dit --debug --batch_split 2 -c latest -d ${preprocessed_data_dir}
```

### 3. テンプレート形状に対する構造化モデル生成
1.Deep Implicit Templatesの学習により生成されるテンプレート形状の点群モデル（`template_downsampled.ply`）を取得します：
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_template_mesh.py -e examples/sofas_dit --debug -c latest
python sample_points.py -f examples/sofas_dit/TrainingMeshes/template.ply -n 2500
```

上記のテンプレート形状の点群モデルに対して構造化モデルを生成します．
構造化モデル生成ネットワークとして，従来手法である[iMG](https://github.com/smiyauchi199/structured_mesh_generator)と
[Voxel2Mesh](https://github.com/cvlab-epfl/voxel2mesh)のうち，より精度が高い前者を使用します．
2つの比較手法（iMG，Voxel2Mesh）はそれぞれ`previous_methods`フォルダに置いています．

また，テンプレート形状に対する構造化モデルの形状復元精度は，入力物体に対する構造化モデルの精度に直結します．
そのため，クラスごとに構造化モデルを複数回生成し，最も精度が高いモデルをテスト・推論時に使用します．


## テスト（構造化モデル生成）
入力点群から計算した符号付き距離場を用いて構造化モデルを生成します．
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python reconstruct_structured_meshes.py -e examples/sofas_dit -c latest --split examples/splits/sv2_sofas_test.json -d ${preprocessed_data_dir} --skip --batch_split 1 --pointcloud
```
オプション：
* `--modelId`：一つの入力点群に対して構造化モデルを生成（**Demo**を参照）
* `--missing`：欠損を含む入力点群を扱う場合
* `--noise`：ノイズを含む入力点群を扱う場合
* `--interpolation`：2つのモデル潜在変数間に対する線形補間
    * `--first_id`：1つ目のモデル番号
    * `--second_id`：2つ目のモデル番号
    * `--num_interpolation`：線形補間数

各オプションの使い方は`reconstruct.sh`を参照してください．

## 評価
4つの評価指標（`Chamfer距離, Earth Mover's距離,Point-to-Mesh距離, Mesh復元率`）を使用可能です．
各評価指標の詳細は論文（4.1節）を参照してください．
また，評価プログラムの詳細なオプション・使用方法については`evaluate.py`及び`eval.sh`を参照してください．

* Chamfer距離の場合
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py -sd ${preprocessed_data_dir}/master_results/proposed/sofa/reconstruction/MeshesWithPoints -td ${preprocessed_data_dir} -s examples/splits/sv2_sofas_test.json --metric chamfer
```
オプション：
* 実験内容（実験名）：
    * `指定なし`（reconstruction）：入力点群に対する構造化モデル生成結果の評価（4.3節）
    * `--missing`（completion）：欠損を含む点群に対する生成結果の評価（4.4.1節）
    * `--noise`（noise）：ノイズを含む点群に対する生成結果の評価（4.4.2節）
    * `--template`（template）：テンプレート構造化モデルの評価（4.2節）
* 手法：
    * `iMG`：比較手法
    * `Voxel2Mesh`：比較手法
    * `proposed`：提案手法
* 評価指標（`--metric`）：
    * `chamfer`：Chamfer距離
    * `emd`：Earth Mover's距離
    * `mesh_acc`：Point-to-Mesh距離
    * `mesh_comp`：Mesh復元率


## その他



## Acknowledgements
* [DeepSDF](https://github.com/facebookresearch/DeepSDF)
* [Deep Implicit Templates](https://github.com/ZhengZerong/DeepImplicitTemplates)
* [iMG](https://github.com/smiyauchi199/structured_mesh_generator)
* [Voxel2Mesh](https://github.com/cvlab-epfl/voxel2mesh)

