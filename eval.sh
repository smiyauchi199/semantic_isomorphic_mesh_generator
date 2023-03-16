
CUDA=${1}
preprocessed_data_dir=/mnt/nas/3DModelDataset/ShapeNet

class_name_list=("bathtub" "bottle" "car" "sofa")
method_list=("iMG" "Voxel2Mesh" "proposed")
metrics1=("chamfer" "mesh_acc")
metrics2=("emd" "mesh_comp")

# experimetnts options
# reconstruction: " "
# completion: "--missing"
# noise: "--noise"
experiments=("reconstruction" " " "completion" "--missing" "noise" "--noise")

for (( i=0; i<${#experiments[@]} ; i+=2 )) ; do
  for method in ${method_list[@]}; do
    for classname in ${class_name_list[@]}; do
      for metric in ${metrics1[@]}; do
        CUDA_VISIBLE_DEVICES=${CUDA} python evaluate.py -sd ${preprocessed_data_dir}/master-results/${method}/${classname}/${experimetnts[i]}/MeshesWithPoints -td ${preprocessed_data_dir} -s examples/splits/sv2_${classname}s_test.json --metric ${metric} ${experiments[i+1]}
      done
      for metric in ${metrics2[@]}; do
        for iter in {0..9} do
          CUDA_VISIBLE_DEVICES=${CUDA} python evaluate.py -sd ${preprocessed_data_dir}/master-results/${method}/${classname}/${experimetnts[i]}/MeshesWithPoints -td ${preprocessed_data_dir} -s examples/splits/sv2_${classname}s_test.json --metric ${metric} --pattern ${iter} ${experiments[i+1]}
        done
      done
    done
  done
done


# evaluate templates

method_list=("iMG" "Voxel2Mesh")
for method in ${method_list[@]}; do
  for classname in ${class_name_list[@]}; do
    for metric in ${metrics1[@]}; do
      CUDA_VISIBLE_DEVICES=${CUDA} python evaluate.py -sd ${preprocessed_data_dir}/master-results/${method}/${classname}/template/Meshes -td ${preprocessed_data_dir}/master-results/Templates/${classname} --metric ${metric} --template
    done
    for metric in ${metrics2[@]}; do
      for iter in {0..9} do
        CUDA_VISIBLE_DEVICES=${CUDA} python evaluate.py -sd ${preprocessed_data_dir}/master-results/${method}/${classname}/template/Meshes -td ${preprocessed_data_dir} -s examples/splits/sv2_${classname}s_test.json --metric ${metric} --pattern ${iter} --template
      done
    done
  done
done
