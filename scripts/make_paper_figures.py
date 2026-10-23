#!/usr/bin/env python3
"""Собрать рисунки статьи из артефактов прогона.

    python3 scripts/make_paper_figures.py results/kaggle/all/fresh/artifacts <out>

Читает metrics_*.json, analysis.json, rm_metrics.json,
base_ranked.npz, rl_grpo_ranked.npz и rl_grpo_samples.json; пишет
fig_*.pdf/.png в <out> (по умолчанию overleaf/figures).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.dpi": 300,
})
BLUE, ORANGE, GREEN, PINK, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6B7780"


def load(art: str, name: str):
    with open(os.path.join(art, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, out: str, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def fig_training(art: str, out: str) -> None:
    tr = load(art, "metrics_train.json")
    h = tr["history"]
    steps = np.array([e["step"] for e in h if "loss" in e])
    loss = np.array([e["loss"] for e in h if "loss" in e])
    ev = [e for e in h if "val_ndcg@10" in e]
    pop = load(art, "metrics_popularity.json")["summary"]["ndcg@10"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(steps, loss, color=GREY, lw=0.5, alpha=0.6, label="по шагам")
    ax[0].plot(steps[len(steps) - len(rolling(loss, 25)):], rolling(loss, 25),
               color=ORANGE, lw=2.2, label="скользящее среднее")
    ax[0].set(xlabel="Шаг обучения", ylabel="Sampled softmax loss")
    ax[0].set_title("(а) Лосс базовой модели", loc="left", fontweight="bold")
    ax[0].legend(frameon=False)
    if ev:
        es = [e["step"] for e in ev]
        ax[1].plot(es, [e["val_ndcg@10"] for e in ev], "-o", color=ORANGE, label="NDCG@10")
        if "val_recall@100" in ev[0]:
            ax[1].plot(es, [e["val_recall@100"] for e in ev], "-s", color=BLUE, label="Recall@100")
    ax[1].axhline(pop, ls="--", color=GREY, lw=1.2)
    ax[1].text(0.05, pop + 0.002, "популярностный baseline, NDCG@10", color=GREY,
               fontsize=9, transform=ax[1].get_yaxis_transform())
    ax[1].set(xlabel="Шаг обучения", ylabel="Метрика на val")
    ax[1].set_title("(б) Качество на валидации", loc="left", fontweight="bold")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_training")


def fig_rm(art: str, out: str) -> None:
    rm = load(art, "rm_metrics.json")
    ens = rm["ensemble"]
    # новый RM пишет like_auc_random/like_auc_hard, старый — like_auc
    like_key = "like_auc_random" if "like_auc_random" in ens else "like_auc"
    names = ["Spearman\n(досмотр)", "AUC\nлайк, случайный"]
    keys = ["ratio_spearman", like_key]
    if "like_auc_hard" in ens:
        names.insert(2, "AUC\nлайк, hard neg")
        keys.insert(2, "like_auc_hard")
    names += ["AUC\nсессия", "AUC\nвозврат"]
    keys += ["session_auc", "return_auc"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(keys))
    vals = [[m[k] for k in keys] for m in rm["models"]] + [[ens[k] for k in keys]]
    labels = [f"модель {m['model']}" for m in rm["models"]] + ["ансамбль"]
    colors = [GREY, "#A8B0B5", "#4F5A61", ORANGE]
    w = 0.2
    for i, (v, lab) in enumerate(zip(vals, labels)):
        ax.bar(x + (i - 1.5) * w, v, w, label=lab, color=colors[i % len(colors)])
    ax.axhline(0.5, ls=":", color=GREY, lw=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("Значение")
    ax.set_title("Качество reward model на отложенном дне", fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    save(fig, out, "fig_rm")


def fig_rl_dynamics(art: str, out: str) -> None:
    h = load(art, "metrics_rl_grpo_train.json")["history"]
    steps = np.arange(len(h))
    reward = np.array([e["reward"] for e in h])
    kl = np.array([e["kl"] for e in h])
    uniq = np.array([e["unique_items"] for e in h])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(steps, reward, color=GREY, lw=0.7, alpha=0.6)
    ax[0].plot(steps[len(steps) - len(rolling(reward, 9)):], rolling(reward, 9),
               color=ORANGE, lw=2.2)
    ax[0].axhline(reward[0], ls=":", color=BLUE)
    ax[0].set(xlabel="Шаг оптимизатора", ylabel="Сумма за 5 шагов")
    ax[0].set_title("(а) награда за роллаут", loc="left", fontweight="bold")
    ax[1].plot(steps, kl, color=GREEN, lw=1.8)
    ax[1].set(xlabel="Шаг оптимизатора", ylabel="KL")
    ax[1].set_title("(б) KL(π‖π_ref)", loc="left", fontweight="bold")
    ax[2].plot(steps, uniq, color=PINK, lw=1.8)
    ax[2].set(xlabel="Шаг оптимизатора", ylabel="Из 5×1024")
    ax[2].set_title("(в) уникальных айтемов", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_rl_dynamics")


def fig_behavior(art: str, out: str) -> None:
    a = load(art, "analysis.json")["models"]
    b, r = a["base"], a["rl"]
    panels = [
        ("Длина трека, с", "mean_track_length", "{:.0f}"),
        ("Jaccard top-10", "jaccard_top10", "{:.3f}"),
        ("Оценка «досмотр»", "rm_ratio_top10", "{:.3f}"),
        ("Оценка «лайк»", "rm_like_top10", "{:.3f}"),
        ("Повтор из истории", "repeat_from_history", "{:.3f}"),
        ("Популярность выдачи", "mean_log_popularity", "{:.2f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (title, key, fmt) in zip(axes.ravel(), panels):
        vb, vr = b[key], r[key]
        ax.bar([0, 1], [vb, vr], color=[BLUE, ORANGE], width=0.6)
        ax.text(0, vb, fmt.format(vb), ha="center", va="bottom", fontsize=10)
        ax.text(1, vr, fmt.format(vr), ha="center", va="bottom", fontsize=10)
        pct = 100 * (vr - vb) / vb if vb else 0.0
        ax.annotate(f"{pct:+.0f}%", xy=(1, vr * 1.12), ha="center", fontsize=10,
                    color=GREEN if pct >= 0 else PINK)
        ax.set_xticks([0, 1], ["база", "RL"])
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(vb, vr) * 1.25)
    fig.tight_layout()
    save(fig, out, "fig_behavior")


def fig_per_user(art: str, out: str) -> None:
    b = np.load(os.path.join(art, "base_ranked.npz"))
    r = np.load(os.path.join(art, "rl_grpo_ranked.npz"))
    bi = {int(u): i for i, u in enumerate(b["uids"])}
    ri = {int(u): i for i, u in enumerate(r["uids"])}
    common = sorted(set(bi) & set(ri))
    rb = np.array([b[f"per_user_recall@100"][bi[u]] for u in common])
    rr = np.array([r[f"per_user_recall@100"][ri[u]] for u in common])
    nb = np.array([b["per_user_ndcg@10"][bi[u]] for u in common])
    nr = np.array([r["per_user_ndcg@10"][ri[u]] for u in common])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(rb, rr, s=8, alpha=0.4, color=GREY)
    ax[0].plot([0, 1], [0, 1], "--", color=ORANGE)
    ax[0].set(xlim=(0, 1.03), ylim=(0, 1.03),
              xlabel="Recall@100 базовой модели", ylabel="Recall@100 после RL")
    ax[0].set_title("(а) по пользователям", loc="left", fontweight="bold")

    for i, (m, lab) in enumerate([(nb, "NDCG@10"), (nr, "RL")]):
        pass
    lo = np.minimum(nb, nr)
    hi = np.maximum(nb, nr)
    worse = int((nr < nb).sum())
    same = int((nr == nb).sum())
    better = int((nr > nb).sum())
    worse_r = int((rr < rb).sum())
    same_r = int((rr == rb).sum())
    better_r = int((rr > rb).sum())
    x = np.arange(2)
    w = 0.26
    ax[1].bar(x - w, [worse, worse_r], w, color=PINK, label="хуже")
    ax[1].bar(x, [same, same_r], w, color=GREY, label="без изменений")
    ax[1].bar(x + w, [better, better_r], w, color=GREEN, label="лучше")
    for xx, vv in zip([x[0] - w, x[0], x[0] + w, x[1] - w, x[1], x[1] + w],
                      [worse, same, better, worse_r, same_r, better_r]):
        ax[1].text(xx, vv, str(vv), ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(x, ["NDCG@10", "Recall@100"])
    ax[1].set_ylabel("Пользователей")
    ax[1].set_title("(б) направление эффекта", loc="left", fontweight="bold")
    ax[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, out, "fig_per_user")


def fig_multistep(art: str, out: str) -> None:
    ms = load(art, "analysis.json")["multistep"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, lab, col in (("base", "база", BLUE), ("rl", "RL", ORANGE)):
        if key not in ms:
            continue
        steps = [s["step"] for s in ms[key]["steps"]]
        ax[0].plot(steps, [s["unique_items"] for s in ms[key]["steps"]], "-o", color=col, label=lab)
        ax[1].plot(steps, [100 * s["repeat_rate"] for s in ms[key]["steps"]], "-s", color=col, label=lab)
    ax[0].set(xlabel="Шаг генерации", ylabel="Уникальных айтемов")
    ax[0].set_title("(а) Разнообразие выдачи", loc="left", fontweight="bold")
    ax[1].set(xlabel="Шаг генерации", ylabel="Повтор истории/генерации, %")
    ax[1].set_title("(б) Доля повторов", loc="left", fontweight="bold")
    for a_ in ax:
        a_.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_multistep")


def fig_leaderboard(art: str, out: str) -> None:
    base = load(art, "metrics_base.json")["summary"]
    files = [f for f in os.listdir(art) if f.startswith("metrics_rl_") and f.endswith(".json")
             and not f.endswith("_train.json")]
    rl = load(art, sorted(files)[0])["summary"]
    pop = load(art, "metrics_popularity.json")["summary"]
    sas = {"ndcg@10": 0.0748, "recall@100": 0.1026}
    names = ["Популярность", "База", "RL", "SASRec\n(опубл.)"]
    nd = [pop["ndcg@10"], base["ndcg@10"], rl["ndcg@10"], sas["ndcg@10"]]
    rc = [pop["recall@100"], base["recall@100"], rl["recall@100"], sas["recall@100"]]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars_nd = ax.bar(x - w / 2, nd, w, color=[GREY, BLUE, ORANGE, "#999999"], label="NDCG@10")
    for xx, a_ in zip(x - w / 2, nd):
        ax.text(xx, a_, f"{a_:.4f}", ha="center", va="bottom", fontsize=9)
    ax.bar(x + w / 2, rc, w, hatch="//", fill=False, edgecolor="k", label="Recall@100")
    for xx, b_ in zip(x + w / 2, rc):
        ax.text(xx, b_, f"{b_:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(nd + rc) * 1.15)
    ax.set_xticks(x, names)
    ax.set_ylabel("Метрика на тестовом дне")
    ax.set_title("Итоговое сравнение на тестовом дне", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_leaderboard")


def main() -> None:
    art = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle/all/fresh/artifacts"
    out = sys.argv[2] if len(sys.argv) > 2 else "overleaf/figures"
    os.makedirs(out, exist_ok=True)
    for fn in (fig_training, fig_rm, fig_rl_dynamics, fig_behavior, fig_per_user,
               fig_multistep, fig_leaderboard):
        fn(art, out)
        print("ok", fn.__name__, flush=True)


if __name__ == "__main__":
    main()
