# training options:
# - xla_vgg16
# - xla_vgg19
# - xla_resnet18_frz0
# - xla_resnet34
# - xla_resnet50

# model_type="xla_vgg16"
# max_epochs=5
# batch_size=512
# num_workers=8
# data_dir=""

# python -m src.train trainer=gpu \
#                     model=$model_type \
#                     data.batch_size=$batch_size \
#                     data.num_workers=$num_workers \
#                     trainer.max_epochs=$max_epochs \

export CUDA_VISIBLE_DEVICES=0
python -m src.train model=xla_vgg19 trainer=gpu trainer.max_epochs=25

export CUDA_VISIBLE_DEVICES=1
python -m src.train model=xla_vgg16 trainer=gpu trainer.max_epochs=25

export CUDA_VISIBLE_DEVICES=2
python -m src.train model=xla_resnet50 trainer=gpu trainer.max_epochs=25