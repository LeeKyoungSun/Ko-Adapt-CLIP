"""
scripts/visualize.py
Baseline 결과 시각화

- 영어/한국어 Recall@k 비교 막대그래프
- 코사인 유사도 분포 (히스토그램)
- 토큰 수 분포 비교
- 성능 격차 히트맵
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional


# 한국어 폰트 설정 (없으면 기본 폰트로 fallback)
def set_korean_font():
    font_candidates = [
        "NanumGothic", "Nanum Gothic", "AppleGothic",
        "Malgun Gothic", "나눔고딕",
    ]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in font_candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            return font
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans (한국어 폰트 없음 - 영문 출력)"


def plot_recall_comparison(
    en_metrics: Dict,
    ko_metrics: Dict,
    output_path: str,
):
    """영어/한국어 Recall@k 비교 막대그래프"""
    font = set_korean_font()

    k_list = [1, 5, 10]
    metrics_i2t = {
        "EN": [en_metrics[f"I2T_Recall@{k}"] for k in k_list],
        "KO": [ko_metrics[f"I2T_Recall@{k}"] for k in k_list],
    }
    metrics_t2i = {
        "EN": [en_metrics[f"T2I_Recall@{k}"] for k in k_list],
        "KO": [ko_metrics[f"T2I_Recall@{k}"] for k in k_list],
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(k_list))
    width = 0.35
    colors = {"EN": "#4C72B0", "KO": "#DD8452"}

    for ax, (task, data), title in zip(
        axes,
        [("I2T", metrics_i2t), ("T2I", metrics_t2i)],
        ["Image → Text (I2T)", "Text → Image (T2I)"]
    ):
        task, data = task, data
        bars_en = ax.bar(x - width/2, data["EN"], width, label="영어 (EN)",
                         color=colors["EN"], alpha=0.85)
        bars_ko = ax.bar(x + width/2, data["KO"], width, label="한국어 (KO)",
                         color=colors["KO"], alpha=0.85)

        # 값 레이블
        for bar in bars_en:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
        for bar in bars_ko:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("k")
        ax.set_ylabel("Recall@k (%)")
        ax.set_title(f"Recall@k 비교 - {title}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"@{k}" for k in k_list])
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("CLIP Baseline: 영어 vs 한국어 Recall@k", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Recall@k 차트 저장: {output_path}")


def plot_cosine_distribution(
    en_sims: np.ndarray,
    ko_sims: np.ndarray,
    output_path: str,
):
    """코사인 유사도 분포 히스토그램"""
    set_korean_font()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(en_sims, bins=50, alpha=0.6, label=f"영어 (μ={en_sims.mean():.3f})",
            color="#4C72B0", density=True)
    ax.hist(ko_sims, bins=50, alpha=0.6, label=f"한국어 (μ={ko_sims.mean():.3f})",
            color="#DD8452", density=True)
    ax.axvline(en_sims.mean(), color="#4C72B0", linestyle="--", linewidth=1.5)
    ax.axvline(ko_sims.mean(), color="#DD8452", linestyle="--", linewidth=1.5)
    ax.set_xlabel("코사인 유사도")
    ax.set_ylabel("밀도")
    ax.set_title("이미지-텍스트 코사인 유사도 분포", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"코사인 분포 차트 저장: {output_path}")


def plot_token_distribution(
    en_counts: list,
    ko_counts: list,
    output_path: str,
):
    """토큰 수 분포 비교"""
    set_korean_font()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, counts, label, color in zip(
        axes,
        [en_counts, ko_counts],
        ["영어 (EN)", "한국어 (KO)"],
        ["#4C72B0", "#DD8452"]
    ):
        arr = np.array(counts)
        ax.hist(arr, bins=40, color=color, alpha=0.8)
        ax.axvline(arr.mean(), color="red", linestyle="--",
                   label=f"평균: {arr.mean():.1f}")
        ax.axvline(77, color="black", linestyle=":", linewidth=2,
                   label="CLIP 한도 (77)")
        ax.set_xlabel("토큰 수")
        ax.set_ylabel("캡션 수")
        ax.set_title(f"토큰 수 분포 - {label}", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("BPE 토크나이저 토큰 수 분포 비교", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"토큰 분포 차트 저장: {output_path}")


def plot_performance_gap_heatmap(
    results: Dict,   # {"en": metrics, "ko": metrics}
    output_path: str,
):
    """성능 격차 히트맵 (10주차 최종 보고서용 포맷)"""
    set_korean_font()

    keys = ["I2T_Recall@1", "I2T_Recall@5", "I2T_Recall@10",
            "T2I_Recall@1", "T2I_Recall@5", "T2I_Recall@10",
            "mean_cosine_similarity"]

    en_vals = [results["en"].get(k, 0) for k in keys]
    ko_vals = [results["ko"].get(k, 0) for k in keys]
    gaps = [ko - en for ko, en in zip(ko_vals, en_vals)]

    data = np.array([en_vals, ko_vals, gaps])
    row_labels = ["영어 (EN)", "한국어 (KO)", "격차 (KO - EN)"]

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        data, annot=True, fmt=".2f",
        xticklabels=keys, yticklabels=row_labels,
        cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 10}
    )
    ax.set_title("CLIP Baseline 성능 격차 히트맵 (EN vs KO)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"히트맵 저장: {output_path}")
