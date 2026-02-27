# preprocessing.py
# 이미지 전처리 담당

from PIL import Image
import numpy as np
import io


def preprocess_image(image_bytes: bytes):
    """
    이미지 바이트 데이터를 모델 입력 형태로 변환

    Args:
        image_bytes: 이미지 파일의 바이트 데이터

    Returns:
        numpy array: (1, 3, 224, 224) 형태의 전처리된 이미지
    """
    # 1. 바이트 데이터를 PIL 이미지로 변환
    image = Image.open(io.BytesIO(image_bytes))

    # 2. RGB로 변환 (PNG 등 알파 채널 제거)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 3. 256으로 리사이즈 후 중앙 224x224 크롭
    image = image.resize((256, 256))
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    # 4. numpy 배열로 변환 및 정규화
    img_array = np.array(image, dtype=np.float32) / 255.0

    # 5. ImageNet 정규화
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # 6. HWC → CHW (ONNX 형식)
    img_array = np.transpose(img_array, (2, 0, 1))

    # 7. 배치 차원 추가 (1, 3, 224, 224)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array