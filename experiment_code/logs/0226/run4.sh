#CUDA_VISIBLE_DEVICES=0,1 python train_ours.py --init_checkpoint_path pretrained_models/Llama2_7b_sharded \
#	--model_config_path pretrained_models/Llama2_7b_hf --wrapped_class_name LlamaDecoderLayer \
#	--data_path datasets/alpaca-train.jsonl --added_tokens 1 \
#	--act_checkpointing --lr 5e-5 --accumulation_steps 16 --batch_size 4 \
#	--checkpoint_path ./checkpoints/self_kd0.1_long_mp0.1 --mask_p 0.1 --hack --self_kd 0.1 --wb_name self_kd0.1_mp0.1_long

CUDA_VISIBLE_DEVICES=0,2 python train_kd.py --init_checkpoint_path pretrained_models/Llama2_7b_sharded \
	--model_config_path pretrained_models/Llama2_7b_hf --wrapped_class_name LlamaDecoderLayer \
	--data_path datasets/alpaca-train.jsonl --prob_path checkpoints/full_prob --added_tokens 1 \
	--act_checkpointing --lr 5e-5 --accumulation_steps 16 --batch_size 4 \
	--checkpoint_path ./checkpoints/prev_kd0.1_mp0.1 --mask_p 0.1 --hack --self_kd 0.1  --wb_name prev_kd0.1_mp0.1

CUDA_VISIBLE_DEVICES=0,2 python train_kd.py --init_checkpoint_path pretrained_models/Llama2_7b_sharded \
	--model_config_path pretrained_models/Llama2_7b_hf --wrapped_class_name LlamaDecoderLayer \
	--data_path datasets/alpaca-train.jsonl --prob_path checkpoints/full_prob --added_tokens 1 \
	--act_checkpointing --lr 5e-5 --accumulation_steps 16 --batch_size 4 \
	--checkpoint_path ./checkpoints/prev_kd0.3 --hack --self_kd 0.3  --wb_name prev_kd0.3

CUDA_VISIBLE_DEVICES=0,2 python train_kd.py --init_checkpoint_path pretrained_models/Llama2_7b_sharded \
	--model_config_path pretrained_models/Llama2_7b_hf --wrapped_class_name LlamaDecoderLayer \
	--data_path datasets/alpaca-train.jsonl --prob_path checkpoints/full_prob --added_tokens 1 \
	--act_checkpointing --lr 5e-5 --accumulation_steps 16 --batch_size 4 \
	--checkpoint_path ./checkpoints/prev_kd0.1_mp0.05 --mask_p 0.05 --hack --self_kd 0.1  --wb_name prev_kd0.1_mp0.05

CUDA_VISIBLE_DEVICES=0,2 python train_kd.py --init_checkpoint_path pretrained_models/Llama2_7b_sharded \
	--model_config_path pretrained_models/Llama2_7b_hf --wrapped_class_name LlamaDecoderLayer \
	--data_path datasets/alpaca-train.jsonl --prob_path checkpoints/full_prob --added_tokens 1 \
	--act_checkpointing --lr 5e-5 --accumulation_steps 16 --batch_size 4 \
	--checkpoint_path ./checkpoints/prev_kd0.1_mp0.2 --mask_p 0.2 --hack --self_kd 0.1  --wb_name prev_kd0.1_mp0.2

