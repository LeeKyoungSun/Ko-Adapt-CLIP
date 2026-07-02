"""
experiments/run_failure_analysis.py
3주차: 실패 유형 분석 실행 스크립트

사용법:
    # 임베딩 새로 추출 + 분석
    python experiments/run_failure_analysis.py

    # 저장된 임베딩 재사용 (빠름)
    python experiments/run_failure_analysis.py --emb_dir outputs/20260702_104729
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

from scripts.data_loader import get_dataloader
from scripts.evaluate import extract_embeddings
from scripts.analyze_failure import analyze_token_vs_similarity, analyze_cultural_keywords

KO_ANNOTATION = str(ROOT / "datasets/kococo/annotations/MSCOCO_train_val_Korean.json")
KO_IMAGE_DIR = str(ROOT / "datasets/kococo/images")


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"failure_analysis_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    #임베딩 로드 또는 추출 
    if args.emb_dir:
        emb_dir = Path(args.emb_dir)
        img_emb_path = emb_dir / "image_embs_ko.npy"
        txt_emb_path = emb_dir / "text_embs_ko.npy"

        if img_emb_path.exists() and txt_emb_path.exists():
            print(f"저장된 임베딩 로드: {emb_dir}")
            image_embs = np.load(img_emb_path)
            text_embs = np.load(txt_emb_path)
        else:
            print(f"임베딩 파일 없음: {emb_dir}")
            print("--emb_dir 없이 다시 실행하거나 --save_embeddings 옵션으로 먼저 저장하세요.")
            return
    else:
        print("임베딩 새로 추출 중...")
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"디바이스: {device}")

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        model.eval()

        loader = get_dataloader(
            image_dir=KO_IMAGE_DIR,
            annotation_file=KO_ANNOTATION,
            transform=preprocess,
            batch_size=64,
            num_workers=4,
            lang="ko",
        )
        embeddings = extract_embeddings(model, loader, device)
        image_embs = embeddings["image_embeddings"]
        text_embs = embeddings["text_embeddings"]

        # 저장
        np.save(out_dir / "image_embs_ko.npy", image_embs)
        np.save(out_dir / "text_embs_ko.npy", text_embs)
        print(f"임베딩 저장 완료: {out_dir}")

    print(f"\n임베딩 shape: 이미지={image_embs.shape}, 텍스트={text_embs.shape}")

    # 분석 1: 토큰 수 vs 코사인 유사도 
    print("\n[분석 1] 토큰 수 vs 코사인 유사도 상관관계")
    token_result = analyze_token_vs_similarity(
        annotation_file=KO_ANNOTATION,
        embeddings_image=image_embs,
        embeddings_text=text_embs,
        output_dir=str(out_dir),
    )
    print(f"피어슨 r: {token_result['pearson_r']}")

    # 분석 2: 한국 고유 개념 키워드 분석
    print("\n[분석 2] 한국 고유 개념 키워드 유사도 분석")
    cultural_result = analyze_cultural_keywords(
        annotation_file=KO_ANNOTATION,
        embeddings_image=image_embs,
        embeddings_text=text_embs,
        output_dir=str(out_dir),
    )

    # 결과 요약
    print("\n" + "=" * 55)
    print(" 3주차 실패 유형 분석 요약")
    print("=" * 55)
    print(f"[토크나이저] 피어슨 r = {token_result['pearson_r']} "
          f"(p={token_result['p_value']:.2e})")
    print(f"[문화 개념]  고유 개념 포함: "
          f"{cultural_result['cultural_vs_general']['cultural_mean_cosine']:.4f} vs "
          f"일반: {cultural_result['cultural_vs_general']['general_mean_cosine']:.4f} "
          f"(격차: {cultural_result['cultural_vs_general']['gap']:.4f})")
    print(f"\n결과 저장 완료: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3주차 실패 유형 분석")
    parser.add_argument(
        "--emb_dir", type=str, default=None,
        help="저장된 임베딩 디렉토리 경로 (없으면 새로 추출)"
    )
    args = parser.parse_args()
    main(args)
