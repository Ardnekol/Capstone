#!/usr/bin/env bash
cd /u/student/2024/cs24mtech11024/Capstone/stage_2/cross_domain_eval
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
echo "GPU $CUDA_VISIBLE_DEVICES  $(date)"
CPY=/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python
echo "===== ZeroWaste-f (region_proposal) ====="
$CPY eval_detection.py --dataset zerowaste --det-method region_proposal \
  --output-dir ../eval_results/detection_zerowaste_rp --device cuda:0
echo "===== WaRP-D (region_proposal) ====="
$CPY eval_detection.py --dataset warpd --det-method region_proposal \
  --output-dir ../eval_results/detection_warpd_rp --device cuda:0
echo "ALL DONE $(date)"
