# Generation of semantic isomorphic mesh using implicit function representation

This repository implements the method proposed in a paper, titled:  
**"Generation of semantic isomorphic mesh using implicit function representation."**


## Citation

If you use this work, please cite the following BibTeX entry:

```
@article{miyauchigeneration,
  title={Generation of Semantic Isomorphic Mesh Using Implicit Function Representation},
  author={Miyauchi, Shoko and Itaya, Kyo and Morooka, Ken'ichi},
  journal = {SSRN: https://ssrn.com/abstract=5205387},
  month = {4},
  year = {2025}
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


## Environment Setup
Install the virtual environment using conda:
```bash
conda env create -f environment.yml
conda activate deepsdf
```

## Demo
Pre-trained weights for the four classes used in the paper (bathtub) are located in the examples directory.
Each class uses its own working directory: examples/{class_name}_dit, which contains training logs and model weights.

Below is a sample command using the bathtub class:
```bash
#** Common variables **
GPU_ID=0
preprocessed_data_dir=/mnt/nas/3DModelDataset/ShapeNet  # Absolute path recommended
#** Common variables **

modelId=example_bathtub
python predict_sdf.py -d ./data -f ${modelId}.ply
CUDA_VISIBLE_DEVICES=${GPU_ID} python reconstruct_structured_meshes.py -d ./data -c latest -m ${modelId} --batch_split 2
```
`Adjust batch_split based on your GPU memory. Smaller values reduce loop iterations and speed up model generation.`

The output semantic isomorphic mesh (SIM) `${modelId}.ply` will be saved in `./data/MeshesWithPoints`.

## Data Preparation & Preprocessing
We use [ShapeNet](https://www.shapenet.org). Download it and place it under `3DModelDataset/ShapeNet`.
Model IDs for training/testing are stored as `.json` files in `examples/splits`.

### SDF from CAD Models
The signed distance field (SDF) data calculated from CAD models are stored under `3DModelDataset/ShapeNet/SdfSamples`,
and the normalization parameters that map each CAD model to a unit sphere are stored under `3DModelDataset/ShapeNet/NormalizationParameters`,
both in `.npz` format.

These SDFs and normalization parameters are generated using [DeepSDF](https://github.com/facebookresearch/DeepSDF).

Run Example:
```bash
# Generate SDF
python preprocess_data.py --data_dir ${preprocessed_data_dir} --source ${preprocessed_data_dir}/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_bathtubs_train.json --skip
# Sample point cloud from surface
python preprocess_data.py --data_dir ${preprocessed_data_dir} --source ${preprocessed_data_dir}/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_bathtubs_train.json --skip --surface
```

### SDF from Point Clouds
Save generated SDFs from input point clouds in `3DModelDataset/ShapeNet/PointCloudSdfSamples`.
The SDFs are generated using `make_sdf_dataset.py`:
```bash
python make_sdf_dataset.py -d ${preprocessed_data_dir} -s examples/splits/sv2_bathtubs_test.json
```
Optional arguments:
* `--missing`: for incomplete point clouds
* `--noise`: for noisy point clouds


## Training

The training data consists of SDFs computed from CAD models using DeepSDF.
The training pipeline includes:
1. [Deep Implicit Templates](https://github.com/ZhengZerong/DeepImplicitTemplates) training
2. Inverse warping function (shape deformation network) training
3. Isomorphic mesh generation from the template shape (Template mesh generation)

### 1. Deep Implicit Templates (DIT) Training
We train [Deep Implicit Templates](https://github.com/ZhengZerong/DeepImplicitTemplates) to obtain the template space.
For network and training parameters, please refer to `examples/bathtubs/specs.json`.
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_deep_implicit_templates.py -e examples/bathtubs_dit --debug --batch_split 2 -c latest -d ${preprocessed_data_dir}
```

#### Checking Training Results
```bash
# Template shape
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_template_mesh.py -e examples/bathtubs_dit --debug 
# Reconstructed shapes (The topology and number of vertices vary for each model.)
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_training_meshes.py -e examples/bathtubs_dit --debug --start_id 0 --end_id 20 --octree --keep_normalization
# Canonical positions in template space
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_canonical_positions.py -e examples/bathtubs_dit --debug --start_id 0 --end_id 20
# Correspondences across all models (All models are color-coded based on their coordinates in the template space.)
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_meshes_correspondence.py -e examples/bathtubs_dit --debug --start_id 0 --end_id 20
```

### 2. Inverse Warping Function Training
We train an inverse warping function that maps from the template space to the input space.
For network and training parameters, please refer to `examples/bathtubs/specs.json`.
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_inversed_function.py -e examples/bathtubs_dit --debug --batch_split 2 -c latest -d ${preprocessed_data_dir}
```

### 3. Isomorphic mesh generation from the template shape (Template mesh generation)
1.Obtain the point cloud model of the template shape (`template_downsampled.ply`) generated through the training of Deep Implicit Templates:
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_template_mesh.py -e examples/bathtubs_dit --debug -c latest
python sample_points.py -f examples/bathtubs_dit/TrainingMeshes/template.ply -n 2500
```

We generate an isomorphic mesh (IM) from the above point cloud model of the template shape.
As the IM generation network, we use the previously proposed method [iMG](https://github.com/smiyauchi199/structured_mesh_generator), 
which achieves higher accuracy than [Voxel2Mesh](https://github.com/cvlab-epfl/voxel2mesh).
Both comparison methods (iMG and Voxel2Mesh) are located in the `previous_methods` folder.

The reconstruction accuracy of the IM for the template shape directly affects the accuracy of the SIMs for the input objects.
Therefore, we generate multiple IMs of the template shape per class and use the one with the highest accuracy during testing and inference.


## Testing（SIM generation）
Use SDFs from point clouds to generate SIMs:
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python reconstruct_structured_meshes.py -e examples/bathtubs_dit -c latest --split examples/splits/sv2_bathtubs_test.json -d ${preprocessed_data_dir} --skip --batch_split 1 --pointcloud
```
Optional arguments:
* `--modelId`：for single point cloud input (see **Demo**)
* `--missing`：for incomplete point clouds
* `--noise`：for noisy point clouds
* `--interpolation`：interpolate between two latent codes
    * `--first_id`：The first model ID
    * `--second_id`：The second model ID
    * `--num_interpolation`：Number of linear interpolations

See `reconstruct.sh` for more options.

## Evaluation
Metrics:
1. Chamfer Distance
2. Earth Mover’s Distance
3. Point-to-Mesh Distance
4. Mesh Completion Rate

See `evaluate.py` and `eval.sh` for usage.

* Chamfer Distance
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py -sd ${preprocessed_data_dir}/master_results/proposed/bathtub/reconstruction/MeshesWithPoints -td ${preprocessed_data_dir} -s examples/splits/sv2_bathtubs_test.json --metric chamfer
```
Options:
* Experiment types:
    * `指定なし`（reconstruction）：Evaluation of SIM Generation Results for Input Point Clouds
    * `--missing`（completion）：Evaluation of SIM Generation Results for Point Clouds with Missing Regions
    * `--noise`（noise）：Evaluation of SIM Generation Results for Point Clouds with Noise
    * `--template`（template）：template mesh evaluation

* Metrics（`--metric`）：
    * `chamfer`：Chamfer Distance
    * `emd`：Earth Mover’s Distance
    * `mesh_acc`：Point-to-Mesh Distance
    * `mesh_comp`：Mesh Completion Rate

## References
* [DeepSDF](https://github.com/facebookresearch/DeepSDF)
* [Deep Implicit Templates](https://github.com/ZhengZerong/DeepImplicitTemplates)
* [iMG](https://github.com/smiyauchi199/structured_mesh_generator)
* [Voxel2Mesh](https://github.com/cvlab-epfl/voxel2mesh)

