export CUDA_VISIBLE_DEVICES=0

images_folder='data/test_character_recognition'

python -m images_folder=$images_folder src.infer checkpoint_path="logs/train_xla/runs/2024-05-23_03-09-03/checkpoints/epoch_067.ckpt" output_path="results/resnet18"
python -m images_folder=$images_folder src.infer checkpoint_path="logs/train_xla/runs/2024-05-23_03-17-11/checkpoints/epoch_061.ckpt" output_path="results/resnet34"
python -m images_folder=$images_folder src.infer checkpoint_path="logs/train_xla/runs/2024-05-23_02-59-48/checkpoints/epoch_055.ckpt" output_path="results/vgg16"
python -m images_folder=$images_folder src.infer checkpoint_path="logs/train_xla/runs/2024-05-23_02-14-44/checkpoints/epoch_069.ckpt" output_path="results/vgg19"

# not use
# python -m src.infer checkpoint_path="logs/train_xla/runs/2024-05-23_03-04-01/checkpoints/epoch_057.ckpt" output_path="results/resnet50"
# python -m src.infer checkpoint_path="logs/train_xla/runs/2024-05-23_16-03-18/checkpoints/epoch_070.ckpt" output_path="results/vgg16_up"

python -m src.ensemble images_folder=$images_folder