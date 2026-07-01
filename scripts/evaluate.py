"""
scripts/evaluate.py
CLIP 임베딩 추출 및 Recall@k / 코사인 유사도 계산
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Dict


def extract_embeddings(model, dataloader, device: str) -> Dict[str, np.ndarray]:
    #이미지와 텍스트 임베딩을 배치 단위로 추출
    model.eval()
    all_image_embs = []
    all_text_embs = []
    all_image_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="임베딩 추출 중"):
            images = batch["image"].to(device)
            captions = batch["caption"]
            image_ids = batch["image_id"]

            # 이미지 임베딩
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            # 텍스트 임베딩
            import open_clip
            tokens = open_clip.tokenize(captions).to(device)
            text_features = model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)

            all_image_embs.append(image_features.cpu().numpy())
            all_text_embs.append(text_features.cpu().numpy())
            all_image_ids.extend(image_ids.tolist())

    return {
        "image_embeddings": np.vstack(all_image_embs),   # (N, D)
        "text_embeddings": np.vstack(all_text_embs),     # (N, D)
        "image_ids": all_image_ids,
    }


def compute_recall_at_k(
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    k_list: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Image-to-Text 및 Text-to-Image Recall@k를 계산합니다.
    (N, D) 임베딩 기준, 대각선이 정답 쌍.
    """
    N = image_embs.shape[0]

    # 코사인 유사도 행렬 (N x N)
    sim_matrix = image_embs @ text_embs.T  # (N, N)

    results = {}

    # Image-to-Text (I2T)
    i2t_ranks = []
    for i in range(N):
        scores = sim_matrix[i]  # i번째 이미지 vs 모든 텍스트
        sorted_idx = np.argsort(-scores)
        rank = np.where(sorted_idx == i)[0][0] + 1  # 1-indexed
        i2t_ranks.append(rank)

    for k in k_list:
        results[f"I2T_Recall@{k}"] = np.mean(np.array(i2t_ranks) <= k) * 100

    # Text-to-Image (T2I)
    t2i_ranks = []
    for i in range(N):
        scores = sim_matrix[:, i]  # 모든 이미지 vs i번째 텍스트
        sorted_idx = np.argsort(-scores)
        rank = np.where(sorted_idx == i)[0][0] + 1
        t2i_ranks.append(rank)

    for k in k_list:
        results[f"T2I_Recall@{k}"] = np.mean(np.array(t2i_ranks) <= k) * 100

    # 평균 코사인 유사도 (대각선)
    diag_sims = sim_matrix.diagonal()
    results["mean_cosine_similarity"] = float(np.mean(diag_sims))
    results["std_cosine_similarity"] = float(np.std(diag_sims))

    return results


def find_hard_negatives(
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    image_ids: List[int],
    bottom_pct: float = 0.2,
) -> List[Dict]:
    """
    성능 하위 20% 사례를 추출합니다. (2주차 샘플링용)
    """
    N = image_embs.shape[0]
    sim_matrix = image_embs @ text_embs.T

    pair_sims = sim_matrix.diagonal()
    threshold = np.percentile(pair_sims, bottom_pct * 100)

    hard_cases = []
    for i in range(N):
        if pair_sims[i] <= threshold:
            hard_cases.append({
                "image_id": image_ids[i],
                "cosine_similarity": float(pair_sims[i]),
                "rank": int(np.sum(sim_matrix[i] >= pair_sims[i])),
            })

    hard_cases.sort(key=lambda x: x["cosine_similarity"])
    print(f"하위 {bottom_pct*100:.0f}% 추출: {len(hard_cases)}개 사례")
    return hard_cases
