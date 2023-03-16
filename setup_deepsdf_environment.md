
# DeepSDFの環境構築
ここでは，[DeepSDF](https://github.com/facebookresearch/DeepSDF)を使用するために必要となる
符号付き距離場のデータを作成するための環境構築方法について説明します．
（DeepSDFのGithubのより詳細な方法や躓いた点を述べます．）


## インストールするライブラリ
* CLI11
* Pangolin
* nanoflann
* Eigen3

インストール作業は仮想環境:**deepsdf**の中で行い，
インストール先として，`/opt`ディレクトリに移動してください：
```bash
conda activate deepsdf
cd /opt
```

以下に順にライブラリのインストールコマンドを記載します．


## CLI11
```bash
git clone https://github.com/CLIUtils/CLI11
cd CLI11/

mkdir build && cd build
git submodule update --init
cmake ..
cmake --build .

sudo cmake --install .
```


## Pangolin
```bash
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin

# Install dependencies (as described above, or your preferred method)
./scripts/install_prerequisites.sh recommended

# ブランチを移動
git checkout v0.6

# Configure and build
mkdir build && cd build
cmake ..
cmake --build .

sudo make install
```


## nanoflann
```bash
git clone https://github.com/jlblancoc/nanoflann.git
cd nanoflann

# Install dependencies (Eigen3もインストールされる)
sudo apt-get install build-essential cmake libgtest-dev libeigen3-dev

mkdir build && cd build && cmake ..
make && make test
```


## DeepSDF
```bash
git clone https://github.com/facebookresearch/DeepSDF.git
cd DeepSDF
cd third-party && rm -r * && git clone https://github.com/rogersce/cnpy.git && cd ..

# fix source file
sudo vi ./src/Utils.h  #include <nanoflann/nanoflann.hpp>  -->  <nanoflann.hpp>

mkdir build && cd build
cmake -DCMAKE_CXX_STANDARD=17 ..
make -j
```

以上の手順により，`/opt/DeepSDF/bin`以下に2つの実行プログラムが生成されます．
* PreprocessMesh：CADモデルから符号付き距離場を計算するプログラム
* SampleVisibleMeshSurface：CADモデルの表面から点群をサンプリングするプログラム
`bin`ディレクトリを`DeepImplicitTemplates-iMG`下にコピーして使用してください．
