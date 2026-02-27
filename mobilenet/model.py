# model.py
# 모델 로드 및 예측 담당

import onnxruntime as ort
import numpy as np
import json
import urllib.request
import os

# 전역 변수 (서버 시작 시 한 번만 로드)
_session = None
_labels = None

# 파일 경로
_model_dir = os.path.dirname(__file__)
_model_path = os.path.join(_model_dir, "mobilenetv2.onnx")
_labels_path = os.path.join(_model_dir, "imagenet_labels.json")


def _download_labels():
    """ImageNet 1000 클래스 라벨 다운로드"""
    if not os.path.exists(_labels_path):
        print("🔄 ImageNet 라벨 다운로드 중...")
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        urllib.request.urlretrieve(url, _labels_path)
        print("✅ 라벨 다운로드 완료!")


def load_model():
    """MobileNetV2 ONNX 모델 로드 (싱글톤 패턴)"""
    global _session, _labels

    if _session is None:
        if not os.path.exists(_model_path):
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {_model_path}\n"
                "먼저 python mobilenet/download_model.py 를 실행하세요."
            )

        print("🔄 MobileNetV2 ONNX 모델 로딩 중...")
        _session = ort.InferenceSession(_model_path)
        print("✅ 모델 로딩 완료!")

        # 라벨 로드
        _download_labels()
        with open(_labels_path, "r") as f:
            _labels = json.load(f)

    return _session, _labels


def predict(processed_image, top_k: int = 5):
    """
    전처리된 이미지로 분류 예측 수행

    Args:
        processed_image: 전처리된 이미지 배열 (1, 3, 224, 224)
        top_k: 반환할 상위 결과 개수

    Returns:
        list: 예측 결과 리스트 [{label, probability}, ...]
    """
    session, labels = load_model()

    # 입력 이름 확인 및 예측 수행
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: processed_image})

    # 소프트맥스로 확률 변환
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    # 상위 K개 결과 추출
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    # 결과 정리
    results = []
    for idx in top_indices:
        results.append({
            "label": labels[idx],
            "probability": round(float(probabilities[idx]) * 100, 1)
        })

    return results