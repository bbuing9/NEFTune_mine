CUDA_VISIBLE_DEVICES=4,5 python train_ours.py --init_checkpoint_path pretrained_models/Llama2_7b_sharded \
	--model_config_path pretrained_models/Llama2_7b_hf --wrapped_class_name LlamaDecoderLayer \
	--data_path datasets/alpaca-train.jsonl --added_tokens 1 \
	--act_checkpointing --lr 5e-5 --accumulation_steps 16 --batch_size 4 \
	--checkpoint_path ./checkpoints/child_d0.5_long_c_cap --child_d --select cap --reverse_p 0.5 --hack --wb_name child_d0.5_long_c_cap
	
CUDA_VISIBLE_DEVICES=4,5 python train_ours.py --init_checkpoint_path pretrained_models/Llama2_7b_sharded \
	--model_config_path pretrained_models/Llama2_7b_hf --wrapped_class_name LlamaDecoderLayer \
	--data_path datasets/alpaca-train.jsonl --added_tokens 1 \
	--act_checkpointing --lr 5e-5 --accumulation_steps 16 --batch_size 4 \
	--checkpoint_path ./checkpoints/child_d0.1_long_c_cap --child_d --select cap --reverse_p 0.1 --hack --wb_name child_d0.1_long_c_cap

CUDA_VISIBLE_DEVICES=4,5 python train_kd.py --init_checkpoint_path pretrained_models/Llama2_7b_sharded \
	--model_config_path pretrained_models/Llama2_7b_hf --wrapped_class_name LlamaDecoderLayer \
	--data_path datasets/alpaca-train.jsonl --added_tokens 1 \
	--act_checkpointing --lr 5e-5 --accumulation_steps 16 --batch_size 4 \
	--checkpoint_path ./checkpoints/child_d0.3_long_c_cap_kd0.1 --child_d --select cap --reverse_p 0.3 --self_kd 0.1 --hack --wb_name child_d0.3_long_c_cap_kd0.1

CUDA_VISIBLE_DEVICES=4,5 python train_kd.py --init_checkpoint_path pretrained_models/Llama2_7b_sharded \
	--model_config_path pretrained_models/Llama2_7b_hf --wrapped_class_name LlamaDecoderLayer \
	--data_path datasets/alpaca-train.jsonl --added_tokens 1 \
	--act_checkpointing --lr 5e-5 --accumulation_steps 16 --batch_size 4 \
	--checkpoint_path ./checkpoints/child_d0.3_long_c_cap_kd0.01 --child_d --select cap --reverse_p 0.3 --self_kd 0.01 --hack --wb_name child_d0.3_long_c_cap_kd0.01