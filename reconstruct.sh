
CUDA=${1}
preprocessed_data_dir=/mnt/nas/3DModelDataset/ShapeNet

class_name_list=("bathtub" "bottle" "car" "sofa")
for name in ${class_name_list[@]};
do
  # reconstruct DIT results
  CUDA_VISIBLE_DEVICES=${CUDA} python reconstruct_deep_implicit_templates.py -e examples/${name}s_dit -c latest --split examples/splits/sv2_${name}s_test.json -d ${preprocessed_data_dir}

  # deepsdf data (preprocessed sdf data with cad models)
  CUDA_VISIBLE_DEVICES=${CUDA} python reconstruct_structured_meshes.py -e examples/${name}s_dit -c latest --split examples/splits/sv2_${name}s_test.json -d ${preprocessed_data_dir} --skip --batch_split 1

  # point cloud
  CUDA_VISIBLE_DEVICES=${CUDA} python reconstruct_structured_meshes.py -e examples/${name}s_dit -c latest --split examples/splits/sv2_${name}s_test.json -d ${preprocessed_data_dir} --skip --batch_split 1 --pointcloud

  # missing point cloud
  CUDA_VISIBLE_DEVICES=${CUDA} python reconstruct_structured_meshes.py -e examples/${name}s_dit -c latest --split examples/splits/sv2_${name}s_test.json -d ${preprocessed_data_dir} --skip --batch_split 1 --pointcloud --missing

  # noised point cloud
  CUDA_VISIBLE_DEVICES=${CUDA} python reconstruct_structured_meshes.py -e examples/${name}s_dit -c latest --split examples/splits/sv2_${name}s_test.json -d ${preprocessed_data_dir} --skip --batch_split 1 --pointcloud --noise

  # interpolate
  CUDA_VISIBLE_DEVICES=${CUDA} python reconstruct_structured_meshes.py -e examples/${name}s_dit -c latest --split examples/splits/sv2_${name}s_test.json -d ${preprocessed_data_dir} --skip --batch_split 1 --pointcloud --interpolation --first_id 0 --second_id 1 --num_interpolation 5
done

