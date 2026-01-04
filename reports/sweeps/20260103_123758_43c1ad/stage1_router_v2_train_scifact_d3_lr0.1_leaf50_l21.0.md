# Stage1 Router v2 Train (scifact d3_lr0.1_leaf50_l21.0)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `200`
- seed: `12`
- feature_cols: `raw_max_top3, abs_logit, logit_sq, logit_sigmoid, logit_cube, cs_ret, top1_sim, top2_sim, top3_sim, delta12, count_sim_ge_t, margin, topk_gap, topk_ratio, topk_entropy, score_span, topk_mean, topk_std`
- auc: `0.7598`
- ece: `0.0312`
- t_accept: `0.8800`
- t_reject: `0.0000`
- fp_accept_rate: `0.0100`
- fn_reject_rate: `0.0000`
- uncertain_rate: `0.7500`
- stage2_route_rate: `0.7500`
- ok_rate: `0.9900`