SAVE_PATH_SHARDED=pretrained_models/Llama2_7b_sharded
SAVE_PATH_HF=pretrained_models/Llama2_7b_hf

python convert_hf_to_fsdp.py --load_path meta-llama/Llama-2-7b-hf \
	--save_path $SAVE_PATH_SHARDED \
	--save_path_hf $SAVE_PATH_HF
