"""
experiments/run_baseline.py
Ko-Adapt CLIP — 영어/한국어 Baseline 측정 및 성능 격차 정량화

사용법:
    python experiments/run_baseline.py --lang both
    python experiments/run_baseline.py --lang both --max_samples 500   # 빠른 테스트
    python experiments/run_baseline.py --lang both --model_name ViT-L-14
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import open_clip

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from scripts.data_loader import get_dataloader
from scripts.evaluate import extract_embeddings, compute_recall_at_k, find_hard_negatives
from scripts.tokenizer_analysis import compare_tokenization
from scripts.visualize import (
    plot_recall_comparison,
    plot_cosine_distribution,
    plot_token_distribution,
    plot_performance_gap_heatmap,
)

#경로설정
PATHS = {
    "en": {
        "image_dir": ROOT / "datasets/mscoco/images",
        "annotation": ROOT / "datasets/mscoco/annotations/captions_val2014.json",
    },
    "ko": {
        "image_dir": ROOT / "datasets/kococo/images",
        "annotation": ROOT / "datasets/kococo/annotations/kococo_val2014.json",
    },
}


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("baseline")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run_single(lang: str, model, preprocess, device: str, args, logger) -> dict:
    logger.info(f"[{lang.upper()}] Baseline 측정 시작")

    loader = get_dataloader(
        image_dir=str(PATHS[lang]["image_dir"]),
        annotation_file=str(PATHS[lang]["annotation"]),
        transform=preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        lang=lang,
    )

    embeddings = extract_embeddings(model, loader, device)

    metrics = compute_recall_at_k(
        embeddings["image_embeddings"],
        embeddings["text_embeddings"],
        k_list=[1, 5, 10],
    )

    hard_negatives = find_hard_negatives(
        embeddings["image_embeddings"],
        embeddings["text_embeddings"],
        embeddings["image_ids"],
        bottom_pct=0.2,
    )

    logger.info(f"[{lang.upper()}] I2T Recall@1={metrics['I2T_Recall@1']:.2f}%  "
                f"Recall@5={metrics['I2T_Recall@5']:.2f}%  "
                f"Recall@10={metrics['I2T_Recall@10']:.2f}%")
    logger.info(f"[{lang.upper()}] 코사인 유사도: {metrics['mean_cosine_similarity']:.4f} "
                f"± {metrics['std_cosine_similarity']:.4f}")

    return {
        "metrics": metrics,
        "hard_negatives": hard_negatives,
        "embeddings": embeddings,
    }


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(ROOT / "logs" / f"baseline_{timestamp}.log")
    logger.info(f"실험 시작 | 모델: {args.model_name} | pretrained: {args.pretrained}")
    logger.info(f"출력 디렉토리: {out_dir}")

    #모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"디바이스: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    model.eval()

    #실험 설정 저장
    config = vars(args)
    config["device"] = device
    config["timestamp"] = timestamp
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    langs = ["en", "ko"] if args.lang == "both" else [args.lang]
    results = {}

    for lang in langs:
        res = run_single(lang, model, preprocess, device, args, logger)
        results[lang] = res

        with open(out_dir / f"metrics_{lang}.json", "w", encoding="utf-8") as f:
            json.dump(res["metrics"], f, ensure_ascii=False, indent=2)

        with open(out_dir / f"hard_negatives_{lang}.json", "w", encoding="utf-8") as f:
            json.dump(res["hard_negatives"], f, ensure_ascii=False, indent=2)

        if args.save_embeddings:
            np.save(out_dir / f"image_embs_{lang}.npy",
                    res["embeddings"]["image_embeddings"])
            np.save(out_dir / f"text_embs_{lang}.npy",
                    res["embeddings"]["text_embeddings"])

    #영어/한국어 비교
    if "en" in results and "ko" in results:
        en_m = results["en"]["metrics"]
        ko_m = results["ko"]["metrics"]

        comparison = {"en": en_m, "ko": ko_m, "gap_ko_minus_en": {}}
        for key in en_m:
            comparison["gap_ko_minus_en"][key] = round(ko_m[key] - en_m[key], 4)

        with open(out_dir / "comparison_en_ko.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info(" 영어 vs 한국어 성능 격차 요약")
        logger.info("=" * 60)
        logger.info(f"{'지표':<30} {'EN':>8} {'KO':>8} {'Gap':>8}")
        logger.info("-" * 60)
        for key in en_m:
            unit = "%" if "Recall" in key else ""
            gap = ko_m[key] - en_m[key]
            logger.info(
                f"{key:<30} {en_m[key]:>7.2f}{unit} {ko_m[key]:>7.2f}{unit} {gap:>+7.2f}{unit}"
            )

        logger.info("\n시각화")

        plot_recall_comparison(
            en_m, ko_m,
            output_path=str(out_dir / "recall_comparison.png")
        )

        en_diag = (results["en"]["embeddings"]["image_embeddings"] *
                   results["en"]["embeddings"]["text_embeddings"]).sum(axis=1)
        ko_diag = (results["ko"]["embeddings"]["image_embeddings"] *
                   results["ko"]["embeddings"]["text_embeddings"]).sum(axis=1)

        plot_cosine_distribution(
            en_diag, ko_diag,
            output_path=str(out_dir / "cosine_distribution.png")
        )

        plot_performance_gap_heatmap(
            {"en": en_m, "ko": ko_m},
            output_path=str(out_dir / "performance_heatmap.png")
        )

    # 토크나이저 분석
    if args.tokenizer_analysis and "en" in langs and "ko" in langs:
        logger.info("\n토크나이저 분석 시작...")
        en_stats, ko_stats, _ = compare_tokenization(
            en_annotation=str(PATHS["en"]["annotation"]),
            ko_annotation=str(PATHS["ko"]["annotation"]),
            output_path=str(out_dir / "tokenizer_analysis.json"),
            max_samples=args.max_samples,
        )
        plot_token_distribution(
            en_stats["token_counts"], ko_stats["token_counts"],
            output_path=str(out_dir / "token_distribution.png")
        )

    logger.info(f"\n모든 결과 저장 완료: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ko-Adapt CLIP Baseline 측정")

    parser.add_argument("--lang", type=str, default="both",
                        choices=["en", "ko", "both"])
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--tokenizer_analysis", action="store_true")

    args = parser.parse_args()
    main(args)
