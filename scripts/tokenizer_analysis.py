"""
scripts/tokenizer_analysis.py
BPE 토크나이저의 한국어 형태소 과분절 분석 (3주차 준비)

- 영어 vs 한국어 캡션의 토큰 수 분포 비교
- CLIP 토큰 한도(77) 초과 비율 측정
- 과분절 상위 사례 추출
"""

import json
import numpy as np
import open_clip
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict


def analyze_tokenization(
    captions: List[str],
    lang: str,
    max_token_len: int = 77,
) -> Dict:
    """
    캡션 리스트의 토크나이징 통계를 반환합니다.
    """
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    token_counts = []
    overflow_cases = []

    for caption in tqdm(captions, desc=f"[{lang}] 토크나이징"):
        tokens = tokenizer([caption])[0]
        # EOS 토큰(0) 전까지 실제 토큰 수 계산
        non_pad = (tokens != 0).sum().item()
        token_counts.append(non_pad)

        if non_pad >= max_token_len:
            overflow_cases.append({
                "caption": caption,
                "token_count": non_pad,
            })

    counts = np.array(token_counts)
    return {
        "lang": lang,
        "n_samples": len(captions),
        "mean_tokens": float(np.mean(counts)),
        "std_tokens": float(np.std(counts)),
        "median_tokens": float(np.median(counts)),
        "max_tokens": int(np.max(counts)),
        "min_tokens": int(np.min(counts)),
        "overflow_count": len(overflow_cases),
        "overflow_rate_pct": len(overflow_cases) / len(captions) * 100,
        "token_counts": counts.tolist(),
        "overflow_cases": overflow_cases[:20],  # 상위 20개만 저장
    }


def load_captions(annotation_file: str, max_samples: int = None, lang: str = "en") -> List[str]:
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        # KoCOCO 형식: [{"file_path": ..., "captions": [...], "caption_ko": [...]}, ...]
        caption_key = "caption_ko" if lang == "ko" else "captions"
        captions = []
        for item in data:
            if "val2014" not in item["file_path"]:
                continue
            item_captions = item.get(caption_key) or []
            if item_captions:
                captions.append(item_captions[0])
    else:
        # 표준 MS-COCO 형식: {"images": [...], "annotations": [...]}
        captions = [ann["caption"] for ann in data["annotations"]]

    if max_samples:
        captions = captions[:max_samples]
    return captions


def compare_tokenization(
    en_annotation: str,
    ko_annotation: str,
    output_path: str,
    max_samples: int = None,
):
    en_captions = load_captions(en_annotation, max_samples, lang="en")
    ko_captions = load_captions(ko_annotation, max_samples, lang="ko")

    en_stats = analyze_tokenization(en_captions, lang="en")
    ko_stats = analyze_tokenization(ko_captions, lang="ko")

    # 분절 비율: 한국어 평균 토큰 수 / 영어 평균 토큰 수
    segmentation_ratio = ko_stats["mean_tokens"] / en_stats["mean_tokens"]

    summary = {
        "english": {k: v for k, v in en_stats.items() if k != "token_counts"},
        "korean": {k: v for k, v in ko_stats.items() if k != "token_counts"},
        "segmentation_ratio_ko_over_en": round(segmentation_ratio, 3),
        "mean_token_gap": round(ko_stats["mean_tokens"] - en_stats["mean_tokens"], 2),
    }

    # 출력
    print("\n" + "=" * 55)
    print(" 토크나이저 분석 결과 (BPE, CLIP ViT-B-32 기준)")
    print("=" * 55)
    print(f"{'지표':<30} {'영어':>10} {'한국어':>10}")
    print("-" * 55)
    print(f"{'평균 토큰 수':<30} {en_stats['mean_tokens']:>10.2f} {ko_stats['mean_tokens']:>10.2f}")
    print(f"{'표준편차':<30} {en_stats['std_tokens']:>10.2f} {ko_stats['std_tokens']:>10.2f}")
    print(f"{'최대 토큰 수':<30} {en_stats['max_tokens']:>10} {ko_stats['max_tokens']:>10}")
    print(f"{'77토큰 초과율':<30} {en_stats['overflow_rate_pct']:>9.2f}% {ko_stats['overflow_rate_pct']:>9.2f}%")
    print(f"{'한/영 분절 비율':<30} {segmentation_ratio:>21.3f}x")
    print("=" * 55)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n토크나이저 분석 저장 완료: {output_path}")

    return en_stats, ko_stats, summary
