"""
scripts/analyze_failure.py
3주차: 실패 유형 분석

1. 토큰 수 vs 코사인 유사도 상관관계
2. 한국 고유 개념 키워드 기반 유사도 분석
3. 실패 사례 시각화
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import open_clip
import torch
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from typing import List, Dict


# ── 한국어 폰트 설정 ─────────────────────────────────────────
def set_korean_font():
    candidates = ["NanumGothic", "Nanum Gothic", "AppleGothic", "Malgun Gothic"]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


# ── 한국 고유 개념 키워드 ────────────────────────────────────
KOREAN_CULTURE_KEYWORDS = {
    "음식": ["김치", "된장", "삼계탕", "불고기", "떡볶이", "김밥", "비빔밥",
             "국밥", "갈비", "순대", "잡채", "냉면", "삼겹살", "쌈"],
    "장소": ["경복궁", "한옥", "절", "사찰", "광화문", "남산", "한강",
             "명동", "인사동", "전주", "제주"],
    "문화": ["한복", "온돌", "장독대", "찜질방", "태권도", "사물놀이",
             "판소리", "탈춤", "소원"],
    "사회": ["눈치", "정", "빨리빨리", "편의점", "치킨"],
}


def get_token_counts(captions: List[str]) -> List[int]:
    """캡션별 토큰 수 반환"""
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    counts = []
    for cap in tqdm(captions, desc="토큰 수 계산"):
        tokens = tokenizer([cap])[0]
        non_pad = (tokens != 0).sum().item()
        counts.append(non_pad)
    return counts


def analyze_token_vs_similarity(
    annotation_file: str,
    embeddings_image: np.ndarray,
    embeddings_text: np.ndarray,
    output_dir: str,
):
    """토큰 수 vs 코사인 유사도 상관관계 분석"""
    set_korean_font()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # val2014만, 첫 번째 캡션만
    captions = []
    for item in data:
        if "val2014" not in item["file_path"]:
            continue
        caps = item.get("caption_ko", [])
        if caps:
            captions.append(caps[0])

    print(f"캡션 수: {len(captions)}")
    assert len(captions) == len(embeddings_image), \
        f"캡션 수({len(captions)})와 임베딩 수({len(embeddings_image)}) 불일치"

    token_counts = get_token_counts(captions)

    # 코사인 유사도 (대각선 = 정답 쌍)
    img_norm = embeddings_image / np.linalg.norm(embeddings_image, axis=1, keepdims=True)
    txt_norm = embeddings_text / np.linalg.norm(embeddings_text, axis=1, keepdims=True)
    cosine_sims = (img_norm * txt_norm).sum(axis=1)

    token_arr = np.array(token_counts)
    sim_arr = np.array(cosine_sims)

    # 피어슨 상관계수
    r, p_value = stats.pearsonr(token_arr, sim_arr)
    print(f"\n피어슨 상관계수: r={r:.4f}, p={p_value:.4e}")

    # ── 구간별 평균 유사도 ────────────────────────────────────
    bins = [0, 20, 30, 40, 50, 60, 77]
    bin_labels = ["~20", "21-30", "31-40", "41-50", "51-60", "61-77"]
    bin_means = []
    bin_stds = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (token_arr >= bins[i]) & (token_arr < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(sim_arr[mask].mean())
            bin_stds.append(sim_arr[mask].std())
            bin_counts.append(mask.sum())
        else:
            bin_means.append(0)
            bin_stds.append(0)
            bin_counts.append(0)

    # ── 시각화 1: 산점도 ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 산점도 (샘플 3000개)
    idx = np.random.choice(len(token_arr), min(3000, len(token_arr)), replace=False)
    axes[0].scatter(token_arr[idx], sim_arr[idx], alpha=0.3, s=5, color="#DD8452")
    z = np.polyfit(token_arr, sim_arr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(token_arr.min(), token_arr.max(), 100)
    axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label=f"추세선 (r={r:.3f})")
    axes[0].axvline(77, color="black", linestyle=":", linewidth=1.5, label="CLIP 한도 (77)")
    axes[0].set_xlabel("토큰 수")
    axes[0].set_ylabel("코사인 유사도")
    axes[0].set_title("토큰 수 vs 코사인 유사도", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 구간별 평균
    x_pos = range(len(bin_labels))
    bars = axes[1].bar(x_pos, bin_means, yerr=bin_stds, capsize=5,
                       color="#4C72B0", alpha=0.8, error_kw={"linewidth": 1.5})
    for i, (bar, cnt) in enumerate(zip(bars, bin_counts)):
        axes[1].text(bar.get_x() + bar.get_width()/2, 0.005,
                     f"n={cnt}", ha="center", va="bottom", fontsize=8, color="white")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"{l}토큰" for l in bin_labels])
    axes[1].set_ylabel("평균 코사인 유사도")
    axes[1].set_title("토큰 수 구간별 평균 유사도", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("BPE 과분절과 성능 저하 관계 분석 (한국어)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "token_vs_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"산점도 저장: {out / 'token_vs_similarity.png'}")

    # 결과 저장
    result = {
        "pearson_r": round(float(r), 4),
        "p_value": float(p_value),
        "n_samples": len(captions),
        "bin_analysis": [
            {
                "range": bin_labels[i],
                "count": int(bin_counts[i]),
                "mean_cosine": round(float(bin_means[i]), 4),
                "std_cosine": round(float(bin_stds[i]), 4),
            }
            for i in range(len(bin_labels))
        ],
    }
    with open(out / "token_similarity_correlation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def analyze_cultural_keywords(
    annotation_file: str,
    embeddings_image: np.ndarray,
    embeddings_text: np.ndarray,
    output_dir: str,
):
    """한국 고유 개념 키워드 포함 캡션의 유사도 분석"""
    set_korean_font()
    out = Path(output_dir)

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions = []
    for item in data:
        if "val2014" not in item["file_path"]:
            continue
        caps = item.get("caption_ko", [])
        captions.append(caps[0] if caps else "")

    img_norm = embeddings_image / np.linalg.norm(embeddings_image, axis=1, keepdims=True)
    txt_norm = embeddings_text / np.linalg.norm(embeddings_text, axis=1, keepdims=True)
    cosine_sims = (img_norm * txt_norm).sum(axis=1)

    # 카테고리별 키워드 포함 캡션 유사도
    results = {}
    all_keyword_indices = set()

    for category, keywords in KOREAN_CULTURE_KEYWORDS.items():
        indices = []
        matched_keywords = []
        for i, cap in enumerate(captions):
            for kw in keywords:
                if kw in cap:
                    indices.append(i)
                    matched_keywords.append(kw)
                    break
        if indices:
            sims = cosine_sims[indices]
            results[category] = {
                "count": len(indices),
                "mean_cosine": float(sims.mean()),
                "std_cosine": float(sims.std()),
                "keywords_found": list(set(matched_keywords)),
            }
            all_keyword_indices.update(indices)

    # 고유 개념 포함 vs 미포함 비교
    mask_cultural = np.zeros(len(captions), dtype=bool)
    mask_cultural[list(all_keyword_indices)] = True
    mask_general = ~mask_cultural

    cultural_mean = cosine_sims[mask_cultural].mean() if mask_cultural.sum() > 0 else 0
    general_mean = cosine_sims[mask_general].mean() if mask_general.sum() > 0 else 0

    print(f"\n한국 고유 개념 포함 캡션: {mask_cultural.sum()}개, 평균 유사도: {cultural_mean:.4f}")
    print(f"일반 캡션: {mask_general.sum()}개, 평균 유사도: {general_mean:.4f}")
    print(f"격차: {cultural_mean - general_mean:.4f}")

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = list(results.keys()) + ["일반 캡션"]
    means = [results[c]["mean_cosine"] for c in results.keys()] + [float(general_mean)]
    stds = [results[c]["std_cosine"] for c in results.keys()] + [float(cosine_sims[mask_general].std())]
    counts = [results[c]["count"] for c in results.keys()] + [int(mask_general.sum())]
    colors = ["#DD8452", "#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    bars = ax.bar(categories, means, yerr=stds, capsize=5,
                  color=colors[:len(categories)], alpha=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"n={cnt}", ha="center", va="bottom", fontsize=9)

    ax.axhline(general_mean, color="gray", linestyle="--", linewidth=1.5,
               label=f"일반 캡션 평균 ({general_mean:.3f})")
    ax.set_ylabel("평균 코사인 유사도")
    ax.set_title("한국 고유 개념 카테고리별 CLIP 유사도", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "cultural_keyword_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"문화 개념 분석 차트 저장: {out / 'cultural_keyword_similarity.png'}")

    final = {
        "cultural_vs_general": {
            "cultural_count": int(mask_cultural.sum()),
            "cultural_mean_cosine": round(float(cultural_mean), 4),
            "general_count": int(mask_general.sum()),
            "general_mean_cosine": round(float(general_mean), 4),
            "gap": round(float(cultural_mean - general_mean), 4),
        },
        "by_category": results,
    }
    with open(out / "cultural_keyword_analysis.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    return final
