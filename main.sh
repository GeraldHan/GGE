# CUDA_VISIBLE_DEVICES=0 python rubi_main.py --dataset cpv2 \
# --mode lmh_rubi \
# --output lmh_rubi_cpv2 \

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset v2 \
# --mode gge_d_bias \
# --debias gradient_sfce \
# --topq 1 \
# --topv -1 \
# --qvp 5 \
# --output gge_d_softmax_v2 \

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset v2 \
# --mode gge_iter \
# --debias gradient_sfce \
# --topq 1 \
# --topv -1 \
# --qvp 5 \
# --output gge_dq_iter_softmax_v2 \

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset v2 \
# --mode gge_tog \
# --debias gradient_sfce \
# --topq 1 \
# --topv -1 \
# --qvp 5 \
# --output gge_dq_tog_softmax_v2 \

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset v2 \
# --mode base \
# --debias none \
# --topq 1 \
# --topv -1 \
# --qvp 5 \
# --output updn_softmax_v2 \

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 \
# --mode gge_iter \
# --debias gradient \
# --topq 1 \
# --topv -1 \
# --qvp 5 \
# --output gge_dq_test \

CUDA_VISIBLE_DEVICES=1 python main.py --dataset v2 \
--mode zero_out \
--debias none \
--topq 1 \
--topv -1 \
--qvp 5 \
--output zero_out_v_only-v2\

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset v2 \
# --mode gge_tog \
# --debias gradient \
# --topq 1 \
# --topv -1 \
# --qvp 5 \
# --output gge_d1f_tog-v2 \


