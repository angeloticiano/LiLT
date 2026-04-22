import torch
import numpy as np
from PIL import Image

# PATCH: substituir detectron2 por PIL + numpy (evita instalar detectron2)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


def load_image(image_path):
    # Ler em BGR (igual ao detectron2.read_image com format="BGR")
    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size
    # Redimensionar para 224x224
    pil_img = pil_img.resize((224, 224), Image.BILINEAR)
    # Converter para numpy BGR
    image_np = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR
    # Tensor CHW
    image = torch.tensor(image_np.copy()).permute(2, 0, 1)
    return image, (w, h)
