"""
experiments/run_fewshot_eval.py
자체 큐레이션 한국 음식 데이터셋 CLIP 유사도 측정

- caption_natural vs caption_translated vs caption_en 세 버전 비교
- 카테고리별 유사도 분석
- clip_hypothesis 검증
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

METADATA_PATH = ROOT / "datasets/fewshot/metadata/fewshot_captions.json"
IMAGE_DIR     = ROOT / "datasets/fewshot/images"


def set_korean_font():
    candidates = ["NanumGothic", "Nanum Gothic", "AppleGothic", "Malgun Gothic"]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


def encode_texts(model, tokenizer, texts, device, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def encode_images(model, preprocess, image_paths, device, batch_size=32):
    all_embs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="이미지 임베딩"):
        batch_paths = image_paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            except Exception as e:
                print(f"이미지 로드 실패: {p} - {e}")
                imgs.append(torch.zeros(3, 224, 224))
        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            emb = model.encode_image(batch_tensor)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def main():
    set_korean_font()

    # ── 디바이스 ─────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"디바이스: {device}")

    # 모델 로드
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    #  메타데이터 로드 
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 이미지 존재 여부 필터링
    valid = []
    missing = []
    for item in metadata:
        p = IMAGE_DIR / item["file_name"]
        if p.exists():
            valid.append(item)
        else:
            missing.append(item["file_name"])

    if missing:
        print(f"이미지 없음 ({len(missing)}개): {missing[:5]}")
    print(f"유효 이미지: {len(valid)}개")

    #  임베딩 추출 
    image_paths = [IMAGE_DIR / item["file_name"] for item in valid]
    image_embs = encode_images(model, preprocess, image_paths, device)

    captions_natural    = [item["caption_natural"]    for item in valid]
    captions_translated = [item["caption_translated"] for item in valid]
    captions_en         = [item["caption_en"]         for item in valid]

    print("텍스트 임베딩 추출 중...")
    emb_natural    = encode_texts(model, tokenizer, captions_natural,    device)
    emb_translated = encode_texts(model, tokenizer, captions_translated, device)
    emb_en         = encode_texts(model, tokenizer, captions_en,         device)

    #  코사인 유사도 계산 
    sim_natural    = (image_embs * emb_natural).sum(axis=1)
    sim_translated = (image_embs * emb_translated).sum(axis=1)
    sim_en         = (image_embs * emb_en).sum(axis=1)

    #  전체 요약 
    print("\n" + "=" * 55)
    print(" 자체 큐레이션 CLIP 유사도 요약")
    print("=" * 55)
    print(f"{'캡션 유형':<20} {'평균':>8} {'표준편차':>8} {'최솟값':>8} {'최댓값':>8}")
    print("-" * 55)
    for label, sims in [
        ("자연어 (KO)", sim_natural),
        ("번역체 (KO)", sim_translated),
        ("영어 (EN)",   sim_en),
    ]:
        print(f"{label:<20} {sims.mean():>8.4f} {sims.std():>8.4f} "
              f"{sims.min():>8.4f} {sims.max():>8.4f}")

    nat_vs_trans = sim_natural.mean() - sim_translated.mean()
    en_vs_nat    = sim_en.mean() - sim_natural.mean()
    print(f"\n자연어 - 번역체 격차: {nat_vs_trans:+.4f}")
    print(f"영어   - 자연어 격차: {en_vs_nat:+.4f}")

    #  카테고리별 분석 
    categories = {}
    for i, item in enumerate(valid):
        cat = item["category"]
        if cat not in categories:
            categories[cat] = {"natural": [], "translated": [], "en": []}
        categories[cat]["natural"].append(sim_natural[i])
        categories[cat]["translated"].append(sim_translated[i])
        categories[cat]["en"].append(sim_en[i])

    print(f"\n{'카테고리':<12} {'n':>4} {'자연어':>8} {'번역체':>8} {'영어':>8} {'자연-번역':>10}")
    print("-" * 55)
    cat_results = []
    for cat, sims in sorted(categories.items()):
        n   = len(sims["natural"])
        nat = np.mean(sims["natural"])
        tra = np.mean(sims["translated"])
        en  = np.mean(sims["en"])
        gap = nat - tra
        cat_results.append((cat, n, nat, tra, en, gap))
        print(f"{cat:<12} {n:>4} {nat:>8.4f} {tra:>8.4f} {en:>8.4f} {gap:>+10.4f}")

    #  출력 디렉토리 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"fewshot_eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 시각화 1: 세 버전 유사도 분포 비교 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 히스토그램
    axes[0].hist(sim_natural,    bins=25, alpha=0.6, label=f"자연어 (μ={sim_natural.mean():.3f})",    color="#DD8452")
    axes[0].hist(sim_translated, bins=25, alpha=0.6, label=f"번역체 (μ={sim_translated.mean():.3f})", color="#4C72B0")
    axes[0].hist(sim_en,         bins=25, alpha=0.6, label=f"영어   (μ={sim_en.mean():.3f})",         color="#55A868")
    axes[0].axvline(sim_natural.mean(),    color="#DD8452", linestyle="--", linewidth=1.5)
    axes[0].axvline(sim_translated.mean(), color="#4C72B0", linestyle="--", linewidth=1.5)
    axes[0].axvline(sim_en.mean(),         color="#55A868", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("코사인 유사도")
    axes[0].set_ylabel("빈도")
    axes[0].set_title("캡션 유형별 유사도 분포", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 평균 막대그래프
    labels = ["자연어\n(KO)", "번역체\n(KO)", "영어\n(EN)"]
    means  = [sim_natural.mean(), sim_translated.mean(), sim_en.mean()]
    stds   = [sim_natural.std(),  sim_translated.std(),  sim_en.std()]
    colors = ["#DD8452", "#4C72B0", "#55A868"]
    bars = axes[1].bar(labels, means, yerr=stds, capsize=6,
                       color=colors, alpha=0.85, width=0.5)
    for bar, mean in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f"{mean:.4f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("평균 코사인 유사도")
    axes[1].set_title("캡션 유형별 평균 유사도", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, max(means) * 1.2)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("자체 큐레이션 데이터셋 CLIP 유사도 비교 (한국 음식 135장)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "caption_type_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 시각화 2: 카테고리별 자연어 vs 번역체 ───────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    cats    = [r[0] for r in cat_results]
    nat_m   = [r[2] for r in cat_results]
    tra_m   = [r[3] for r in cat_results]
    en_m    = [r[4] for r in cat_results]
    x = np.arange(len(cats))
    w = 0.25

    ax.bar(x - w, nat_m, w, label="자연어 (KO)", color="#DD8452", alpha=0.85)
    ax.bar(x,     tra_m, w, label="번역체 (KO)", color="#4C72B0", alpha=0.85)
    ax.bar(x + w, en_m,  w, label="영어   (EN)", color="#55A868", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("평균 코사인 유사도")
    ax.set_title("카테고리별 캡션 유형 비교", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "category_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 시각화 3: 개별 이미지 자연어 vs 번역체 산점도 ───────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sim_translated, sim_natural, alpha=0.5, s=20, color="#DD8452")
    mn = min(sim_translated.min(), sim_natural.min()) - 0.01
    mx = max(sim_translated.max(), sim_natural.max()) + 0.01
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y=x (동일)")
    above = (sim_natural > sim_translated).sum()
    below = (sim_natural < sim_translated).sum()
    ax.set_xlabel("번역체 유사도")
    ax.set_ylabel("자연어 유사도")
    ax.set_title("이미지별 자연어 vs 번역체 유사도", fontsize=12, fontweight="bold")
    ax.legend()
    ax.text(0.05, 0.95, f"자연어 > 번역체: {above}개\n번역체 > 자연어: {below}개",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "natural_vs_translated_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── JSON 저장 ────────────────────────────────────────────
    results = {
        "n_samples": len(valid),
        "overall": {
            "natural":    {"mean": round(float(sim_natural.mean()),    4),
                           "std":  round(float(sim_natural.std()),     4)},
            "translated": {"mean": round(float(sim_translated.mean()), 4),
                           "std":  round(float(sim_translated.std()),  4)},
            "en":         {"mean": round(float(sim_en.mean()),         4),
                           "std":  round(float(sim_en.std()),          4)},
            "gap_natural_vs_translated": round(float(nat_vs_trans), 4),
            "gap_en_vs_natural":         round(float(en_vs_nat),    4),
        },
        "by_category": {
            cat: {
                "n": n,
                "natural_mean":    round(float(nat), 4),
                "translated_mean": round(float(tra), 4),
                "en_mean":         round(float(en),  4),
                "gap_nat_vs_tra":  round(float(gap), 4),
            }
            for cat, n, nat, tra, en, gap in cat_results
        },
        "per_image": [
            {
                "file_name":        item["file_name"],
                "concept":          item["concept"],
                "category":         item["category"],
                "sim_natural":      round(float(sim_natural[i]),    4),
                "sim_translated":   round(float(sim_translated[i]), 4),
                "sim_en":           round(float(sim_en[i]),         4),
                "gap_nat_vs_tra":   round(float(sim_natural[i] - sim_translated[i]), 4),
                "clip_hypothesis":  item.get("clip_hypothesis", ""),
            }
            for i, item in enumerate(valid)
        ],
    }

    with open(out_dir / "fewshot_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료: {out_dir}")
    print("생성된 파일:")
    print("  - caption_type_comparison.png")
    print("  - category_comparison.png")
    print("  - natural_vs_translated_scatter.png")
    print("  - fewshot_eval_results.json")


if __name__ == "__main__":
    main()