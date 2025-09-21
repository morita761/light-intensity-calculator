import cv2
import numpy as np

def rgb_to_hsv_opencv(r, g, b):
    # OpenCVはBGR順なので、RGBをBGRに変換
    bgr_color = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
    return hsv_color[0][0]

# Photoshopで確認したCyanのRGB値を入力
r_cyan, g_cyan, b_cyan = 0, 160, 233
h_cyan, s_cyan, v_cyan = rgb_to_hsv_opencv(r_cyan, g_cyan, b_cyan)
print(f"Cyan (R:{r_cyan}, G:{g_cyan}, B:{b_cyan}) -> HSV (H:{h_cyan}, S:{s_cyan}, V:{v_cyan}) (OpenCV scale)")

# Photoshopで確認したMagentaのRGB値を入力
r_magenta, g_magenta, b_magenta = 228, 0, 127
h_magenta, s_magenta, v_magenta = rgb_to_hsv_opencv(r_magenta, g_magenta, b_magenta)
print(f"Magenta (R:{r_magenta}, G:{g_magenta}, B:{b_magenta}) -> HSV (H:{h_magenta}, S:{s_magenta}, V:{v_magenta}) (OpenCV scale)")

# 例: 赤色の確認
r_red, g_red, b_red = 255, 0, 0
h_red, s_red, v_red = rgb_to_hsv_opencv(r_red, g_red, b_red)
print(f"Red (R:{r_red}, G:{g_red}, B:{b_red}) -> HSV (H:{h_red}, S:{s_red}, V:{v_red}) (OpenCV scale)")