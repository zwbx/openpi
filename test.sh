# terminal 1
cd SimplerEnv
bash scripts/run_pi05.sh  

# terminal 2
uv run scripts/serve_policy.py policy:checkpoint     --policy.config=pi05_simpler_zscore   --policy.dir=/mnt/hdfs/wenbo/vl
a/pi05_simpler_ckpt/pi05_simpler_zscore_32card/80000/