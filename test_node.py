"""
빠른 동작 검증 스크립트 (ComfyUI 없이 실행 가능)
실행: python test_node.py
"""

import torch
from quaternion_grayscale_node import QuaternionGrayscaleNode


def make_image(r, g, b):
    """단색 1x1 배치 이미지 생성"""
    t = torch.tensor([[[[r, g, b]]]], dtype=torch.float32)  # [1,1,1,3]
    return t


node = QuaternionGrayscaleNode()

test_cases = [
    ("순수 빨강",  1.0, 0.0, 0.0),
    ("순수 초록",  0.0, 1.0, 0.0),
    ("순수 파랑",  0.0, 0.0, 1.0),
    ("흰색",       1.0, 1.0, 1.0),
    ("검정",       0.0, 0.0, 0.0),
    ("중간 회색",  0.5, 0.5, 0.5),
    ("임의 색상",  0.8, 0.3, 0.6),
]

print(f"{'픽셀':<14} {'입력 RGB':>20}   {'magnitude':>10} {'projection':>10} {'sandwich':>10}")
print("-" * 72)

for name, r, g, b in test_cases:
    img = make_image(r, g, b)
    results = {}
    for method in ["magnitude", "projection", "sandwich"]:
        out = node.convert(img, method)[0]          # [1,1,1,3]
        results[method] = out[0, 0, 0, 0].item()   # 첫 채널 값

    print(
        f"{name:<14} ({r:.1f}, {g:.1f}, {b:.1f})   "
        f"{results['magnitude']:>10.4f} "
        f"{results['projection']:>10.4f} "
        f"{results['sandwich']:>10.4f}"
    )

# 출력 형태 확인
img_batch = torch.rand(4, 64, 64, 3)
out_batch  = node.convert(img_batch, "magnitude")[0]
assert out_batch.shape == (4, 64, 64, 3), f"shape 오류: {out_batch.shape}"
assert out_batch.min() >= 0.0 and out_batch.max() <= 1.0, "범위 오류"

print("\n배치 shape 검증 통과:", out_batch.shape)
print("모든 테스트 완료.")
