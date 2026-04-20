HF_HOME=/mnt/localdisk/vllm_omni_hf_cache/ \
VLLM_OMNI_STORAGE_PATH=/mnt/localdisk/vllm_omni_storage \
MODEL=/home/claudio/marey_checkpoints/epoch5-global_step70000 \
MOONVALLEY_AI_PATH=/home/yizhu/code/moonvalley_ai_master \
FLOW_SHIFT=0 \
bash examples/online_serving/marey/run_server.sh
