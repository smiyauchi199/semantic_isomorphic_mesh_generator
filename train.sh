
CUDA=${1}
preprocessed_data_dir=/mnt/nas/3DModelDataset/ShapeNet

class_name_list=("bathtub" "bottle" "car" "sofa")
for name in ${class_name_list[@]}; 
do
  # DIT
  CUDA_VISIBLE_DEVICES=${CUDA} python train_deep_implicit_templates.py -e examples/${name}s_dit --debug --batch_split 1 -c latest -d ${preprocessed_data_dir}

  # inverse warping function
  CUDA_VISIBLE_DEVICES=${CUDA} python train_inversed_function.py -e examples/${name}s_dit --debug --batch_split 1 -c latest -d ${preprocessed_data_dir} --continue latest

done

