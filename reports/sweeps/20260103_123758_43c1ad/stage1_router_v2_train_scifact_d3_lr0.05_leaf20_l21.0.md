# Stage1 Router v2 Train (scifact d3_lr0.05_leaf20_l21.0)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `200`
- seed: `12`
- feature_cols: `raw_max_top3, abs_logit, logit_sq, logit_sigmoid, logit_cube, cs_ret, top1_sim, top2_sim, top3_sim, delta12, count_sim_ge_t, margin, topk_gap, topk_ratio, topk_entropy, score_span, topk_mean, topk_std`
- auc: `0.8737`
- ece: `0.1271`
- t_accept: `0.8600`
- t_reject: `0.5900`
- fp_accept_rate: `0.0100`
- fn_reject_rate: `0.0050`
- uncertain_rate: `0.6800`
- stage2_route_rate: `0.6800`
- ok_rate: `0.9850`