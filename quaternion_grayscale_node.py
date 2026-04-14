"""
ComfyUI Custom Node: Quaternion Grayscale
=========================================
색상 이미지의 각 픽셀 (R, G, B) 를 순수 쿼터니언 q = Ri + Gj + Bk 로 표현한 뒤
쿼터니언 연산으로 그레이스케일 값을 도출합니다.

변환 방식 3가지
---------------
magnitude  : |q| = sqrt(R² + G² + B²) / sqrt(3)   — 쿼터니언 노름
projection : (R + G + B) / 3                         — 회색 단위축 (i+j+k)/√3 에 스칼라 투영
sandwich   : µ * q * µ̄  샌드위치 곱 후 벡터 크기   — 쿼터니언 회전
"""

import math
import torch


# ---------------------------------------------------------------------------
# 쿼터니언 유틸리티 (모두 batch 텐서 연산)
# ---------------------------------------------------------------------------

def quat_mul(a: tuple, b: tuple) -> tuple:
    """두 쿼터니언의 곱  (스칼라, i, j, k) * (스칼라, i, j, k)"""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return (
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    )


def quat_conj(q: tuple) -> tuple:
    """켤레 쿼터니언  (w, -x, -y, -z)"""
    w, x, y, z = q
    return (w, -x, -y, -z)


# ---------------------------------------------------------------------------
# 노드 클래스
# ---------------------------------------------------------------------------

class QuaternionGrayscaleNode:
    """색상 이미지 → 쿼터니언 기반 그레이스케일 이미지"""

    METHODS = ["magnitude", "projection", "sandwich"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (cls.METHODS, {"default": "magnitude"}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("grayscale_image",)
    FUNCTION      = "convert"
    CATEGORY      = "image/quaternion"

    # ------------------------------------------------------------------

    def convert(self, image: torch.Tensor, method: str) -> tuple:
        """
        Parameters
        ----------
        image  : [B, H, W, C]  float32, 값 범위 [0, 1]
        method : "magnitude" | "projection" | "sandwich"

        Returns
        -------
        (grayscale_image,)  [B, H, W, 3]  float32, 값 범위 [0, 1]
        """
        if image.shape[-1] < 3:
            # 채널이 3 미만이면 그대로 반환
            return (image,)

        r = image[..., 0]   # [B, H, W]
        g = image[..., 1]
        b = image[..., 2]

        if method == "magnitude":
            gray = self._method_magnitude(r, g, b)
        elif method == "projection":
            gray = self._method_projection(r, g, b)
        elif method == "sandwich":
            gray = self._method_sandwich(r, g, b)
        else:
            raise ValueError(f"알 수 없는 method: {method!r}")

        gray = torch.clamp(gray, 0.0, 1.0)          # [B, H, W]
        gray = gray.unsqueeze(-1).expand(-1, -1, -1, 3)  # [B, H, W, 3]
        return (gray,)

    # ------------------------------------------------------------------
    # 변환 방식 구현
    # ------------------------------------------------------------------

    @staticmethod
    def _method_magnitude(r, g, b) -> torch.Tensor:
        """
        쿼터니언 크기(노름)
        q = Ri + Gj + Bk
        gray = ||q|| / sqrt(3)  — 최댓값 sqrt(3) 으로 나눠 [0,1] 정규화
        """
        norm = torch.sqrt(r ** 2 + g ** 2 + b ** 2)
        return norm / math.sqrt(3.0)

    @staticmethod
    def _method_projection(r, g, b) -> torch.Tensor:
        """
        회색 단위 쿼터니언 µ = (i+j+k)/√3 에 대한 스칼라 투영
        <q, µ> = (R + G + B) / sqrt(3)
        → [0,1] 로 정규화하면  (R + G + B) / 3
        """
        return (r + g + b) / 3.0

    @staticmethod
    def _method_sandwich(r, g, b) -> torch.Tensor:
        """
        쿼터니언 회전 샌드위치 곱
        회전 쿼터니언: µ = cos(θ/2) + sin(θ/2) * n̂   (n̂ = (1,1,1)/√3, θ = π/2)
        변환: q' = µ * q * µ̄
        gray = ||vector_part(q')|| / sqrt(3)

        θ = π/2 → 회색 축 기준 90° 회전으로 색 채널을 균등하게 혼합
        """
        theta = math.pi / 2.0
        half  = theta / 2.0
        cos_h = math.cos(half)                     # 스칼라
        sin_h = math.sin(half) / math.sqrt(3.0)    # 각 축 성분

        # µ = (cos_h, sin_h, sin_h, sin_h)  — 상수 쿼터니언
        mu  = (cos_h, sin_h, sin_h, sin_h)
        mu_ = quat_conj(mu)                        # 역시 상수

        # 순수 쿼터니언 q = (0, R, G, B)  — 텐서
        q = (torch.zeros_like(r), r, g, b)

        # µ * q
        mq = quat_mul(mu, q)

        # (µ * q) * µ̄
        result = quat_mul(mq, mu_)

        # 벡터 부분의 크기
        _, rx, ry, rz = result
        norm = torch.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        return norm / math.sqrt(3.0)


# ---------------------------------------------------------------------------
# ComfyUI 등록
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "QuaternionGrayscale": QuaternionGrayscaleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuaternionGrayscale": "Quaternion Grayscale",
}
