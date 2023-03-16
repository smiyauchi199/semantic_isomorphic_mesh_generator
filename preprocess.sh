
preprocessed_data_dir=/mnt/nas/3DModelDataset/ShapeNet

class_name_list=("bathtub" "bottle" "car" "sofa")

for name in ${class_name_list[@]};
do
  # 学習データ
  python preprocess_data.py --data_dir data --source ${preprocessed_data_dir}/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_${name}s_train.json --skip
  # python preprocess_data.py --data_dir data --source ${preprocessed_data_dir}/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_${name}s_train.json --skip --surface

  # テストデータ
  python preprocess_data.py --data_dir data --source ${preprocessed_data_dir}/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_${name}s_test.json --skip
  # python preprocess_data.py --data_dir data --source ${preprocessed_data_dir}/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_${name}s_test.json --skip --surface
done
