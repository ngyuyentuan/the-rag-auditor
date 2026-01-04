import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class RouterV2:
    def __init__(self, model, feature_cols, logit_col, prob_flip=False):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.logit_col = logit_col
        self.t_accept = None
        self.t_reject = None
        self.prob_flip = bool(prob_flip)

    def predict_proba(self, df):
        x = df[self.feature_cols].to_numpy()
        p = self.model.predict_proba(x)[:, 1]
        if self.prob_flip:
            return 1.0 - p
        return p


def decide_router(p_hat, t_accept, t_reject):
    if t_accept <= t_reject:
        raise ValueError("router thresholds must satisfy t_accept > t_reject")
    if p_hat >= t_accept:
        return "ACCEPT"
    if p_hat <= t_reject:
        return "REJECT"
    return "UNCERTAIN"


def build_feature_frame(df, logit_col, feature_cols):
    out = pd.DataFrame(index=df.index)
    x = df[logit_col].astype(float)
    out[logit_col] = x
    if "abs_logit" in feature_cols:
        out["abs_logit"] = x.abs()
    if "logit_sq" in feature_cols:
        out["logit_sq"] = x * x
    if "logit_sigmoid" in feature_cols:
        out["logit_sigmoid"] = 1.0 / (1.0 + np.exp(-x))
    if "logit_cube" in feature_cols:
        out["logit_cube"] = x * x * x
    if "cs_ret" in feature_cols:
        if "cs_ret" in df.columns:
            out["cs_ret"] = df["cs_ret"].astype(float)
        else:
            out["cs_ret"] = 1.0 / (1.0 + np.exp(-x))
    if "top1_sim" in feature_cols:
        if "raw_max_top3" in df.columns:
            out["top1_sim"] = df["raw_max_top3"].astype(float)
        else:
            out["top1_sim"] = x
    if "top2_sim" in feature_cols:
        if "gap12" in df.columns and "raw_max_top3" in df.columns:
            out["top2_sim"] = (df["raw_max_top3"] - df["gap12"]).astype(float)
        elif "top1_sim" in out.columns and "margin" in out.columns:
            out["top2_sim"] = out["top1_sim"] - out["margin"]
        else:
            out["top2_sim"] = out["top1_sim"] if "top1_sim" in out.columns else x
    if "top3_sim" in feature_cols:
        if "mean_top3" in df.columns:
            out["top3_sim"] = df["mean_top3"].astype(float)
        else:
            out["top3_sim"] = out["top2_sim"] if "top2_sim" in out.columns else x
    if "delta12" in feature_cols:
        if "gap12" in df.columns:
            out["delta12"] = df["gap12"].astype(float)
        elif "top1_sim" in out.columns and "top2_sim" in out.columns:
            out["delta12"] = out["top1_sim"] - out["top2_sim"]
    if "count_sim_ge_t" in feature_cols:
        t = 0.5
        sims = []
        if "top1_sim" in out.columns:
            sims.append(out["top1_sim"])
        if "top2_sim" in out.columns:
            sims.append(out["top2_sim"])
        if "top3_sim" in out.columns:
            sims.append(out["top3_sim"])
        if sims:
            count = (sims[0] >= t).astype(int)
            for s in sims[1:]:
                count = count + (s >= t).astype(int)
            out["count_sim_ge_t"] = count.astype(float)
    if "top1" in feature_cols and "top1" in df.columns:
        out["top1"] = df["top1"].astype(float)
    if "top2" in feature_cols and "top2" in df.columns:
        out["top2"] = df["top2"].astype(float)
    if "top3" in feature_cols and "top3" in df.columns:
        out["top3"] = df["top3"].astype(float)
    if "margin" in feature_cols:
        if "gap12" in df.columns:
            out["margin"] = df["gap12"].astype(float)
        elif "top1" in out.columns and "top2" in out.columns:
            out["margin"] = out["top1"] - out["top2"]
    if "topk_gap" in feature_cols:
        if "top1" in out.columns and "top2" in out.columns:
            out["topk_gap"] = out["top1"] - out["top2"]
        elif "margin" in out.columns:
            out["topk_gap"] = out["margin"]
        else:
            out["topk_gap"] = 0.0
    if "topk_ratio" in feature_cols:
        if "top1" in out.columns and "top2" in out.columns:
            out["topk_ratio"] = out["top1"] / (out["top2"] + 1e-6)
        else:
            out["topk_ratio"] = 0.0
    if "topk_entropy" in feature_cols:
        if "top1" in out.columns and "top2" in out.columns and "top3" in out.columns:
            logits = np.stack([out["top1"], out["top2"], out["top3"]], axis=1)
            logits = logits - np.max(logits, axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            out["topk_entropy"] = -np.sum(probs * np.log(probs + 1e-9), axis=1)
        else:
            out["topk_entropy"] = 0.0
    if "score_span" in feature_cols:
        if "top1" in out.columns and "top2" in out.columns and "top3" in out.columns:
            out["score_span"] = out[["top1", "top2", "top3"]].max(axis=1) - out[["top1", "top2", "top3"]].min(axis=1)
        else:
            out["score_span"] = 0.0
    if "topk_mean" in feature_cols:
        if "mean_top3" in df.columns:
            out["topk_mean"] = df["mean_top3"].astype(float)
        elif "mean_top5" in df.columns:
            out["topk_mean"] = df["mean_top5"].astype(float)
    if "topk_std" in feature_cols:
        if "std_top3" in df.columns:
            out["topk_std"] = df["std_top3"].astype(float)
        elif "std_top5" in df.columns:
            out["topk_std"] = df["std_top5"].astype(float)
    for col in feature_cols:
        if col not in out.columns and col in df.columns:
            out[col] = df[col].astype(float)
    return out[feature_cols]


def fit(df, feature_cols, y_col, logit_col, class_weight=None, prob_flip=False):
    x = df[feature_cols].to_numpy()
    y = df[y_col].astype(int).to_numpy()
    model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight=class_weight)
    model.fit(x, y)
    return RouterV2(model, feature_cols, logit_col, prob_flip=prob_flip)


def save(router, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": router.model,
        "feature_cols": router.feature_cols,
        "logit_col": router.logit_col,
        "t_accept": router.t_accept,
        "t_reject": router.t_reject,
        "prob_flip": router.prob_flip,
    }, path)


def load(path):
    data = joblib.load(Path(path))
    router = RouterV2(data["model"], data["feature_cols"], data["logit_col"], prob_flip=data.get("prob_flip", False))
    router.t_accept = data.get("t_accept")
    router.t_reject = data.get("t_reject")
    return router
