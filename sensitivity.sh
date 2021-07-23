CUDA_VISIBLE_DEVICES=1 python sensitivity.py --dataset cpv2 \
--mode updn \
--debias none \
--load_checkpoint_path logs/gge_d1f_iter \
--output analysis/gge_d1f_iter \

CUDA_VISIBLE_DEVICES=1 python sensitivity.py --dataset cpv2 \
--mode updn \
--debias none \
--load_checkpoint_path logs/gge_q_iter_sf \
--output analysis/gge_q_iter_sf \

CUDA_VISIBLE_DEVICES=1 python sensitivity.py --dataset cpv2 \
--mode updn \
--debias none \
--load_checkpoint_path logs/gge_q_tog_sf-2 \
--output analysis/gge_q_tog_sf \