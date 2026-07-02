"""
experiments/run_week4_analysis.py
4주차: 번역체 심화 분석 + Named Entity 인식 오류 측정

사용법:
    python experiments/run_week4_analysis.py --emb_dir outputs/20260702_123346
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import open_clip

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from scripts.named_entity_analysis import analyze_ne

KO_ANNOTATION = str(ROOT / "datasets/kococo/annotations/MSCOCO_train_val_Korean.json")
KO_IMAGE_DIR  = str(ROOT / "datasets/kococo/images")


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"week4_analysis_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 임베딩 로드 ──────────────────────────────────────────
    emb_dir = Path(args.emb_dir)
    img_path = emb_dir / "image_embs_ko.npy"
    txt_path = emb_dir / "text_embs_ko.npy"

    if not img_path.exists() or not txt_path.exists():
        print(f"임베딩 파일 없음: {emb_dir}")
        print("먼저 아래 명령어로 임베딩을 저장하세요:")
        print("  python experiments/run_baseline.py --lang ko --save_embeddings")
        return

    print(f"임베딩 로드: {emb_dir}")
    image_embs = np.load(img_path)
    text_embs  = np.load(txt_path)
    print(f"임베딩 shape: {image_embs.shape}")

    # image_ids 로드 (저장된 경우) 또는 순서 기반 재구성
    import json as _json
    ann_path = Path(KO_ANNOTATION)
    with open(ann_path, "r", encoding="utf-8") as f:
        data = _json.load(f)

    image_ids = []
    if isinstance(data, list):
        for item in data:
            if "val2014" not in item["file_path"]:
                continue
            image_ids.append(item["id"])
    image_ids = image_ids[:len(image_embs)]
    print(f"image_ids 재구성: {len(image_ids)}개")

    # ── Named Entity 분석 ────────────────────────────────────
    print("\n" + "=" * 55)
    print(" [분석] Named Entity 인식 오류 측정")
    print("=" * 55)

    ne_result = analyze_ne(
        annotation_file=KO_ANNOTATION,
        image_embs=image_embs,
        text_embs=text_embs,
        image_ids=image_ids,
        output_dir=str(out_dir),
    )

    # ── 최종 요약 ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" 4주차 분석 최종 요약")
    print("=" * 55)
    s = ne_result["summary"]
    print(f"NE 포함 캡션:  {s['ne_count']}개 / 전체 {s['total_captions']}개")
    print(f"NE 평균 유사도: {s['ne_mean_cosine']:.4f}")
    print(f"일반 평균 유사도: {s['gen_mean_cosine']:.4f}")
    print(f"격차: {s['gap_ne_vs_general']:+.4f}")
    print(f"\n결과 저장 완료: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4주차 NE 분석")
    parser.add_argument(
        "--emb_dir", type=str, required=True,
        help="저장된 KO 임베딩 디렉토리 (image_embs_ko.npy, text_embs_ko.npy)"
    )
    args = parser.parse_args()
    main(args)
