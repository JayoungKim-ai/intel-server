# model.py
# TFLite 경량 모델 로드 및 예측 담당
# 기존 TensorFlow 전체 모델 대비 메모리 사용량 대폭 감소 (200~300MB → 50~80MB)

import numpy as np
import tensorflow as tf
import json
import os

_interpreter = None
_class_index = None


def load_model():
    """TFLite 모델 로드 (싱글톤 패턴)"""
    global _interpreter, _class_index

    if _interpreter is None:
        print("🔄 MobileNetV2 TFLite 모델 로딩 중...")

        # TFLite 모델 파일 경로 (model.py와 같은 폴더에 저장)
        model_path = os.path.join(os.path.dirname(__file__), "mobilenet_v2.tflite")

        # TFLite 모델이 없으면 Keras 모델에서 변환 후 저장
        if not os.path.exists(model_path):
            print("⏳ 최초 실행: TFLite 모델 변환 중 (1회만 실행됩니다)...")
            from tensorflow.keras.applications import MobileNetV2
            model = MobileNetV2(weights='imagenet')
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(model_path, 'wb') as f:
                f.write(tflite_model)
            del model  # 원본 모델 메모리 해제
            print("✅ TFLite 변환 완료!")

        # TFLite 인터프리터 생성
        _interpreter = tf.lite.Interpreter(model_path=model_path)
        _interpreter.allocate_tensors()

        # ImageNet 클래스 인덱스 로드 (1000개 클래스명)
        class_index_path = tf.keras.utils.get_file(
            'imagenet_class_index.json',
            'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        )
        with open(class_index_path) as f:
            _class_index = json.load(f)

        print("✅ TFLite 모델 로딩 완료!")

    return _interpreter, _class_index


def predict(processed_image, top_k: int = 5):
    """
    전처리된 이미지로 분류 예측 수행 (TFLite 버전)

    Args:
        processed_image: 전처리된 이미지 배열 (1, 224, 224, 3)
        top_k: 반환할 상위 결과 개수

    Returns:
        list: 예측 결과 리스트 [{label, probability}, ...]
    """
    interpreter, class_index = load_model()

    # 입출력 텐서 정보
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 입력 데이터 설정
    interpreter.set_tensor(
        input_details[0]['index'],
        processed_image.astype(np.float32)
    )

    # 추론 실행
    interpreter.invoke()

    # 결과 가져오기
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # 상위 K개 결과 추출
    top_indices = predictions.argsort()[-top_k:][::-1]

    results = []
    for i in top_indices:
        label = class_index[str(i)][1]
        probability = float(predictions[i]) * 100
        results.append({
            "label": label,
            "probability": round(probability, 1)
        })

    return results