"""
scripts/named_entity_analysis.py
4주차: Named Entity 기반 한국 고유명사 인식 오류 측정

- 한국 장소명, 인명, 브랜드, 문화재 등 NE 카테고리별 필터링
- NE 포함 캡션 vs 일반 캡션 유사도 비교
- 고유명사별 개별 유사도 측정
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import List, Dict, Tuple


# ── 한국어 폰트 ──────────────────────────────────────────────
def set_korean_font():
    candidates = ["NanumGothic", "Nanum Gothic", "AppleGothic", "Malgun Gothic"]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


# ── Named Entity 사전 ────────────────────────────────────────
NAMED_ENTITIES = {
    "장소_서울": [
        "경복궁", "창덕궁", "덕수궁", "광화문", "남산", "한강",
        "명동", "인사동", "홍대", "강남", "이태원", "종로",
        "북촌", "서촌", "청계천", "동대문", "남대문",
    ],
    "장소_지방": [
        "전주", "경주", "부산", "제주", "안동", "강릉",
        "속초", "여수", "통영", "순천", "광주", "대구",
        "전주 한옥마을", "해운대", "오죽헌", "불국사", "석굴암",
    ],
    "문화재": [
        "첨성대", "거북선", "팔만대장경", "수원화성", "종묘",
        "창경궁", "경회루", "근정전", "석가탑", "다보탑",
    ],
    "인명": [
        "이순신", "세종대왕", "김구", "유관순", "안중근",
        "정약용", "신사임당", "이황", "이이",
    ],
    "브랜드/기관": [
        "삼성", "현대", "LG", "롯데", "SK", "카카오", "네이버",
        "코레일", "서울대", "연세대", "고려대",
    ],
    "음식_고유명사": [
        "비빔밥", "삼계탕", "불고기", "갈비", "떡볶이",
        "냉면", "설렁탕", "순대", "막걸리", "소주",
        "삼겹살", "치킨", "김치찌개", "된장찌개", "부대찌개",
    ],
}

# 전체 NE 플랫 리스트
ALL_NE = {ne: cat for cat, nes in NAMED_ENTITIES.items() for ne in nes}


def load_captions_with_ids(annotation_file: str) -> List[Dict]:
    """KoCOCO 포맷에서 val2014 캡션 로드"""
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    if isinstance(data, list):
        for item in data:
            if "val2014" not in item["file_path"]:
                continue
            caps = item.get("caption_ko", [])
            if caps:
                items.append({
                    "image_id": item["id"],
                    "file_path": item["file_path"],
                    "caption": caps[0],
                })
    return items


def find_ne_in_captions(items: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """NE 포함/미포함 캡션 분리"""
    ne_items = []
    general_items = []

    for item in items:
        cap = item["caption"]
        found = []
        for ne, cat in ALL_NE.items():
            if ne in cap:
                found.append({"ne": ne, "category": cat})

        if found:
            item["named_entities"] = found
            ne_items.append(item)
        else:
            item["named_entities"] = []
            general_items.append(item)

    return ne_items, general_items


def compute_similarities(
    items: List[Dict],
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    id2idx: Dict[int, int],
) -> List[float]:
    """image_id 기준으로 코사인 유사도 계산"""
    sims = []
    for item in items:
        idx = id2idx.get(item["image_id"])
        if idx is not None:
            sim = float((image_embs[idx] * text_embs[idx]).sum())
            item["cosine_similarity"] = sim
            sims.append(sim)
    return sims


def analyze_ne(
    annotation_file: str,
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    image_ids: List[int],
    output_dir: str,
):
    set_korean_font()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # id → 임베딩 인덱스 매핑
    id2idx = {iid: i for i, iid in enumerate(image_ids)}

    # 캡션 로드 및 NE 필터링
    all_items = load_captions_with_ids(annotation_file)
    ne_items, general_items = find_ne_in_captions(all_items)

    print(f"전체 캡션: {len(all_items)}개")
    print(f"NE 포함:   {len(ne_items)}개 ({len(ne_items)/len(all_items)*100:.1f}%)")
    print(f"일반:      {len(general_items)}개")

    # 유사도 계산
    ne_sims      = compute_similarities(ne_items,      image_embs, text_embs, id2idx)
    general_sims = compute_similarities(general_items, image_embs, text_embs, id2idx)

    ne_arr  = np.array(ne_sims)
    gen_arr = np.array(general_sims)

    print(f"\nNE 포함 평균 유사도:  {ne_arr.mean():.4f} ± {ne_arr.std():.4f}")
    print(f"일반 캡션 평균 유사도: {gen_arr.mean():.4f} ± {gen_arr.std():.4f}")
    print(f"격차 (NE - 일반):     {ne_arr.mean() - gen_arr.mean():+.4f}")

    # ── 카테고리별 분석 ──────────────────────────────────────
    cat_results = {}
    for item in ne_items:
        if "cosine_similarity" not in item:
            continue
        for ne_info in item["named_entities"]:
            cat = ne_info["category"]
            if cat not in cat_results:
                cat_results[cat] = []
            cat_results[cat].append(item["cosine_similarity"])

    print(f"\n{'카테고리':<20} {'n':>5} {'평균 유사도':>12} {'일반 대비':>10}")
    print("-" * 52)
    cat_summary = []
    for cat, sims in sorted(cat_results.items()):
        arr = np.array(sims)
        gap = arr.mean() - gen_arr.mean()
        cat_summary.append((cat, len(sims), arr.mean(), arr.std(), gap))
        print(f"{cat:<20} {len(sims):>5} {arr.mean():>12.4f} {gap:>+10.4f}")

    # ── 개별 NE별 분석 ───────────────────────────────────────
    ne_word_results = {}
    for item in ne_items:
        if "cosine_similarity" not in item:
            continue
        for ne_info in item["named_entities"]:
            ne = ne_info["ne"]
            if ne not in ne_word_results:
                ne_word_results[ne] = []
            ne_word_results[ne].append(item["cosine_similarity"])

    # 최소 3개 이상 샘플 있는 NE만
    valid_ne = {ne: sims for ne, sims in ne_word_results.items() if len(sims) >= 3}
    print(f"\n개별 NE (샘플 3개 이상): {len(valid_ne)}개")

    # 하위 10개 (가장 유사도 낮은 NE)
    ne_sorted = sorted(valid_ne.items(), key=lambda x: np.mean(x[1]))
    print(f"\n유사도 하위 NE 10개:")
    print(f"{'고유명사':<15} {'n':>4} {'평균 유사도':>12}")
    print("-" * 35)
    for ne, sims in ne_sorted[:10]:
        print(f"{ne:<15} {len(sims):>4} {np.mean(sims):>12.4f}")

    # ── 시각화 1: NE 포함 vs 일반 분포 ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(gen_arr, bins=40, alpha=0.6, color="#4C72B0",
                 label=f"일반 캡션 (μ={gen_arr.mean():.4f})", density=True)
    axes[0].hist(ne_arr,  bins=40, alpha=0.6, color="#DD8452",
                 label=f"NE 포함 (μ={ne_arr.mean():.4f})",   density=True)
    axes[0].axvline(gen_arr.mean(), color="#4C72B0", linestyle="--", linewidth=1.5)
    axes[0].axvline(ne_arr.mean(),  color="#DD8452", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("코사인 유사도")
    axes[0].set_ylabel("밀도")
    axes[0].set_title("NE 포함 vs 일반 캡션 유사도 분포", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 카테고리별 막대
    cats   = [r[0] for r in cat_summary]
    c_mean = [r[2] for r in cat_summary]
    c_std  = [r[3] for r in cat_summary]
    x = range(len(cats))
    axes[1].bar(x, c_mean, yerr=c_std, capsize=4, color="#DD8452", alpha=0.8)
    axes[1].axhline(gen_arr.mean(), color="#4C72B0", linestyle="--",
                    linewidth=1.5, label=f"일반 평균 ({gen_arr.mean():.4f})")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("평균 코사인 유사도")
    axes[1].set_title("NE 카테고리별 평균 유사도", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Named Entity 인식 오류 분석", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "ne_similarity_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nNE 분포 차트 저장: {out / 'ne_similarity_analysis.png'}")

    # ── 시각화 2: 개별 NE 유사도 (샘플 3개 이상) ────────────
    if valid_ne:
        ne_names = [ne for ne, _ in ne_sorted]
        ne_means = [np.mean(sims) for _, sims in ne_sorted]
        ne_counts = [len(sims) for _, sims in ne_sorted]

        fig, ax = plt.subplots(figsize=(max(10, len(ne_names) * 0.6), 6))
        colors = ["#C44E52" if m < gen_arr.mean() else "#55A868" for m in ne_means]
        bars = ax.bar(range(len(ne_names)), ne_means, color=colors, alpha=0.85)
        ax.axhline(gen_arr.mean(), color="#4C72B0", linestyle="--",
                   linewidth=1.5, label=f"일반 평균 ({gen_arr.mean():.4f})")
        for i, (bar, cnt) in enumerate(zip(bars, ne_counts)):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.0005,
                    f"n={cnt}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(ne_names)))
        ax.set_xticklabels(ne_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("평균 코사인 유사도")
        ax.set_title("고유명사별 CLIP 유사도 (빨강=일반 이하, 초록=일반 이상)",
                     fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "ne_word_similarity.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"NE 개별 차트 저장: {out / 'ne_word_similarity.png'}")

    # ── JSON 저장 ─────────────────────────────────────────────
    result = {
        "summary": {
            "total_captions":   len(all_items),
            "ne_count":         len(ne_items),
            "general_count":    len(general_items),
            "ne_mean_cosine":   round(float(ne_arr.mean()),  4),
            "gen_mean_cosine":  round(float(gen_arr.mean()), 4),
            "gap_ne_vs_general": round(float(ne_arr.mean() - gen_arr.mean()), 4),
        },
        "by_category": {
            cat: {
                "count":       n,
                "mean_cosine": round(float(mean), 4),
                "std_cosine":  round(float(std),  4),
                "gap_vs_general": round(float(gap), 4),
            }
            for cat, n, mean, std, gap in cat_summary
        },
        "by_ne_word": {
            ne: {
                "count":       len(sims),
                "mean_cosine": round(float(np.mean(sims)), 4),
                "gap_vs_general": round(float(np.mean(sims) - gen_arr.mean()), 4),
            }
            for ne, sims in ne_word_results.items()
        },
    }
    with open(out / "ne_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"JSON 저장: {out / 'ne_analysis.json'}")

    return result
