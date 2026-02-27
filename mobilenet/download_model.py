# 모델 다운로드 스크립트 (한 번만 실행)
import urllib.request
import os

url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
save_path = os.path.join(os.path.dirname(__file__), "mobilenetv2.onnx")

if not os.path.exists(save_path):
    print("🔄 MobileNetV2 ONNX 모델 다운로드 중...")
    urllib.request.urlretrieve(url, save_path)
    print(f"✅ 다운로드 완료: {save_path}")
else:
    print("✅ 모델 파일이 이미 존재합니다.")