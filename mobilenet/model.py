# model.py
# 모델 로드 및 예측 담당

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions


# 모델을 전역 변수로 선언 (서버 시작 시 한 번만 로드)
_model = None


def load_model():
    """MobileNetV2 모델 로드 (싱글톤 패턴)"""
    global _model
    
    if _model is None:
        print("🔄 MobileNetV2 모델 로딩 중...")
        _model = MobileNetV2(weights='imagenet')
        print("✅ 모델 로딩 완료!")
    
    return _model


def predict(processed_image, top_k: int = 5):
    """
    전처리된 이미지로 분류 예측 수행
    
    Args:
        processed_image: 전처리된 이미지 배열 (1, 224, 224, 3)
        top_k: 반환할 상위 결과 개수
        
    Returns:
        list: 예측 결과 리스트 [{label, probability}, ...]
    """
    # 모델 가져오기
    model = load_model()
    
    # 예측 수행
    predictions = model.predict(processed_image)
    
    # 결과 디코딩 (ImageNet 클래스명으로 변환)
    decoded = decode_predictions(predictions, top=top_k)[0]
    
    # 결과 정리
    results = []
    for (class_id, label, probability) in decoded:
        results.append({
            "label": label,
            "probability": round(float(probability) * 100, 1)
        })
    
    return results