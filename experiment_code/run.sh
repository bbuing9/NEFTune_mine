CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --init_checkpoint_path pretrained_models/Llama2_7b_sharded \
	--model_config_path pretrained_models/Llama2_7b_hf --wrapped_class_name LlamaDecoderLayer \
	--data_path datasets/alpaca-train.jsonl --added_tokens 1 \
	--act_checkpointing --lr 5e-5 --accumulation_steps 8 --batch_size 4 \
	--checkpoint_path ./checkpoints/linear --linear --hack --wb_name linear

