HF_HOME=/mnt/localdisk/vllm_omni_hf_cache \
VLLM_OMNI_STORAGE_PATH=/mnt/localdisk/vllm_omni_storage \
MODEL=/app/hf_checkpoints/marey-distilled-0100 \
MOONVALLEY_AI_PATH=/home/yizhu/code/moonvalley_ai_master \
bash examples/online_serving/marey/run_server.sh
