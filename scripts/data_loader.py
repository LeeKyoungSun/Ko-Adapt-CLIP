"""
scripts/data_loader.py
MS-COCO / KoCOCO 데이터 로더
"""

import os
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class COCOCaptionDataset(Dataset):
    """
    MS-COCO 또는 KoCOCO 캡션 데이터셋 로더.
    image_id 기준으로 이미지와 캡션을 페어링합니다.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform=None,
        max_samples: Optional[int] = None,
        lang: str = "en",  # "en" or "ko"
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.lang = lang

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.pairs = []

        if isinstance(data, list):
            # KoCOCO 형식: [{"file_path": ..., "id": ..., "captions": [...], "caption_ko": [...]}, ...]
            caption_key = "caption_ko" if lang == "ko" else "captions"
            for item in data:
                if "val2014" not in item["file_path"]:
                    continue
                captions = item.get(caption_key) or []
                if not captions:
                    continue
                self.pairs.append({
                    "image_id": item["id"],
                    "file_name": item["file_path"],
                    "caption": captions[0],
                })
        else:
            # 표준 MS-COCO 형식: {"images": [...], "annotations": [...]}
            id2file = {img["id"]: img["file_name"] for img in data["images"]}

            # 이미지별 첫 번째 캡션만 사용 (Recall@k 평가 기준)
            seen = set()
            for ann in data["annotations"]:
                iid = ann["image_id"]
                if iid not in seen:
                    seen.add(iid)
                    self.pairs.append({
                        "image_id": iid,
                        "file_name": id2file[iid],
                        "caption": ann["caption"],
                    })

        if max_samples:
            self.pairs = self.pairs[:max_samples]

        print(f"[{lang.upper()}] 로드 완료: {len(self.pairs)}개 이미지-캡션 쌍")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        image_path = self.image_dir / item["file_name"]

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"이미지 없음: {image_path}")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "caption": item["caption"],
            "image_id": item["image_id"],
        }


def get_dataloader(
    image_dir: str,
    annotation_file: str,
    transform=None,
    batch_size: int = 64,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    lang: str = "en",
) -> DataLoader:
    dataset = COCOCaptionDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=transform,
        max_samples=max_samples,
        lang=lang,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    import open_clip

    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

    en_loader = get_dataloader(
        image_dir="datasets/mscoco/images/val2014",
        annotation_file="datasets/mscoco/annotations/captions_val2014.json",
        transform=preprocess, batch_size=4, max_samples=10, num_workers=0, lang="en",
    )
    ko_loader = get_dataloader(
        image_dir="datasets/kococo/images",
        annotation_file="datasets/kococo/annotations/MSCOCO_train_val_Korean.json",
        transform=preprocess, batch_size=4, max_samples=10, num_workers=0, lang="ko",
    )

    en_batch = next(iter(en_loader))
    ko_batch = next(iter(ko_loader))
    print("EN 이미지:", en_batch["image"].shape)
    print("EN 캡션:", en_batch["caption"][0])
    print("KO 이미지:", ko_batch["image"].shape)
    print("KO 캡션:", ko_batch["caption"][0])
