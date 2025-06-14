#!/usr/bin/env python3
"""
完整化妝應用程序
將所有化妝效果應用到人臉上
包括：眼妝、唇妝、腮紅、修容、眉毛、高光等
"""

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional

class FullMakeupApplication:
    def __init__(self):
        """初始化完整化妝應用"""
        print("💄 初始化完整化妝應用程序")
        
        # MediaPipe 初始化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 完整的面部區域關鍵點定義 - 基於官方MediaPipe文檔
        self.FACE_REGIONS = {
            # 眼部區域 - 使用官方MediaPipe標準
            'left_eye': [
                263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466
            ],
            'right_eye': [
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
            ],
            'left_eye_upper': [
                263, 249, 390, 373, 374, 380, 381, 382
            ],
            'right_eye_upper': [
                33, 7, 163, 144, 145, 153, 154, 155
            ],
            'left_eye_lower': [
                362, 398, 384, 385, 386, 387, 388, 466
            ],
            'right_eye_lower': [
                133, 173, 157, 158, 159, 160, 161, 246
            ],
            
            # 眉毛區域 - 使用官方MediaPipe標準
            'left_eyebrow': [
                300, 293, 334, 296, 336, 285, 295, 282, 283, 276
            ],
            'right_eyebrow': [
                70, 63, 105, 66, 107, 55, 65, 52, 53, 46
            ],
            
            # 嘴唇區域 - 使用官方MediaPipe標準
            'lips': [
                # Outer lips
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
                # Inner lips  
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191
            ],
            'upper_lip': [
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
            ],
            'lower_lip': [
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191
            ],
            
            # 鼻子區域 - 使用官方MediaPipe標準
            'nose': [
                1, 2, 94, 247, 275, 340, 344, 345, 346, 347, 348, 349, 350, 49, 279, 19, 20, 94, 125
            ],
            'nose_bridge': [
                6, 168, 8, 9, 10, 151
            ],
            'nose_tip': [
                1, 2, 5, 4, 19, 94, 125
            ],
            
            # 臉頰區域（腮紅）
            'left_cheek': [
                116, 117, 118, 119, 120, 121, 128, 126, 142, 36, 205, 206, 207, 213, 192, 147
            ],
            'right_cheek': [
                345, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 454, 323, 361, 340
            ],
            
            # 額頭區域（高光）
            'forehead': [
                9, 10, 151, 337, 299, 333, 298, 301, 368, 264, 447, 366, 401, 435, 410, 454
            ],
            
            # 下巴區域（修容）
            'chin': [
                18, 175, 199, 175, 18, 175, 199, 175, 18, 175, 199, 175
            ],
            
            # 臉部輪廓 - 使用官方MediaPipe Silhouette標準
            'face_oval': [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
                377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
        }
        
        # 完整的化妝配色方案 - 基於官方標準RGB顏色，轉換為BGR格式 (OpenCV格式)
        self.MAKEUP_COLORS = {
            # 眼妝 - 基於專業化妝品顏色標準 (BGR格式)
            'eyeshadow_brown': (33, 67, 101),     # 標準棕色眼影 BGR(33,67,101) from RGB(101,67,33)
            'eyeshadow_gold': (0, 215, 255),     # 標準金色眼影 BGR(0,215,255) from RGB(255,215,0)
            'eyeshadow_pink': (193, 182, 255),   # 標準粉色眼影 BGR(193,182,255) from RGB(255,182,193)
            'eyeshadow_smoky': (105, 105, 105),  # 標準煙燻色 BGR(105,105,105) from RGB(105,105,105)
            'eyeshadow_bronze': (50, 127, 205),  # 青銅色眼影 BGR(50,127,205) from RGB(205,127,50)
            'eyeshadow_neutral': (140, 180, 210), # 中性色眼影 BGR(140,180,210) from RGB(210,180,140)
            'eyeliner_black': (0, 0, 0),         # 標準黑色眼線 BGR(0,0,0) from RGB(0,0,0)
            'eyeliner_brown': (19, 69, 139),     # 標準棕色眼線 BGR(19,69,139) from RGB(139,69,19)
            
            # 唇妝 - 基於經典唇膏顏色 (BGR格式)
            'lipstick_red': (0, 0, 255),         # 經典紅色 BGR(0,0,255) from RGB(255,0,0)
            'lipstick_pink': (203, 192, 255),    # 標準粉色 BGR(203,192,255) from RGB(255,192,203)
            'lipstick_coral': (80, 127, 255),    # 珊瑚色 BGR(80,127,255) from RGB(255,127,80)
            'lipstick_nude': (135, 184, 222),    # 裸色 BGR(135,184,222) from RGB(222,184,135)
            'lipstick_berry': (60, 20, 220),     # 莓果色 BGR(60,20,220) from RGB(220,20,60)
            'lipstick_rose': (225, 228, 255),    # 玫瑰色 BGR(225,228,255) from RGB(255,228,225)
            'lipstick_wine': (32, 0, 128),       # 酒紅色 BGR(32,0,128) from RGB(128,0,32)
            
            # 眉毛 - 自然眉色 (BGR格式)
            'eyebrow_brown': (19, 69, 139),      # 標準棕色 BGR(19,69,139) from RGB(139,69,19)
            'eyebrow_black': (0, 0, 0),          # 標準黑色 BGR(0,0,0) from RGB(0,0,0)
            'eyebrow_taupe': (139, 61, 72),      # 灰棕色 BGR(139,61,72) from RGB(72,61,139)
            
            # 腮紅 - 自然膚色 (BGR格式)
            'blush_pink': (193, 182, 255),       # 粉色腮紅 BGR(193,182,255) from RGB(255,182,193)
            'blush_peach': (185, 218, 255),      # 桃色腮紅 BGR(185,218,255) from RGB(255,218,185)
            'blush_coral': (80, 127, 255),       # 珊瑚色腮紅 BGR(80,127,255) from RGB(255,127,80)
            'blush_rose': (225, 228, 255),       # 玫瑰色腮紅 BGR(225,228,255) from RGB(255,228,225)
            
            # 修容和高光 - 專業修容色 (BGR格式)
            'contour_brown': (45, 82, 160),      # 修容色 BGR(45,82,160) from RGB(160,82,45)
            'contour_taupe': (140, 180, 210),    # 灰棕修容 BGR(140,180,210) from RGB(210,180,140)
            'highlight_champagne': (230, 240, 250), # 香檳高光 BGR(230,240,250) from RGB(250,240,230)
            'highlight_gold': (0, 215, 255),     # 金色高光 BGR(0,215,255) from RGB(255,215,0)
            'highlight_pearl': (245, 240, 255),  # 珍珠高光 BGR(245,240,255) from RGB(255,240,245)
            
            # 鼻影和修容 (BGR格式)
            'nose_contour': (45, 82, 160),       # 鼻影 BGR(45,82,160) from RGB(160,82,45)
            'nose_highlight': (245, 240, 255),   # 鼻樑高光 BGR(245,240,255) from RGB(255,240,245)
        }
        
        # 預設化妝風格
        self.MAKEUP_STYLES = {
            'natural_daily': {
                'name': '自然日常妝',
                'effects': [
                    ('eyeshadow_brown', 'left_eye_upper', 0.3),
                    ('eyeshadow_brown', 'right_eye_upper', 0.3),
                    ('eyebrow_brown', 'left_eyebrow', 0.4),
                    ('eyebrow_brown', 'right_eyebrow', 0.4),
                    ('lipstick_nude', 'lips', 0.5),
                    ('blush_peach', 'left_cheek', 0.2),
                    ('blush_peach', 'right_cheek', 0.2),
                ]
            },
            'glamorous_evening': {
                'name': '魅力晚妝',
                'effects': [
                    ('eyeshadow_smoky', 'left_eye', 0.6),
                    ('eyeshadow_smoky', 'right_eye', 0.6),
                    ('eyeshadow_gold', 'left_eye_upper', 0.4),
                    ('eyeshadow_gold', 'right_eye_upper', 0.4),
                    ('eyeliner_black', 'left_eye_upper', 0.7),
                    ('eyeliner_black', 'right_eye_upper', 0.7),
                    ('eyebrow_black', 'left_eyebrow', 0.5),
                    ('eyebrow_black', 'right_eyebrow', 0.5),
                    ('lipstick_red', 'lips', 0.7),
                    ('blush_coral', 'left_cheek', 0.3),
                    ('blush_coral', 'right_cheek', 0.3),
                    ('highlight_gold', 'forehead', 0.2),
                    ('contour_brown', 'nose', 0.3),
                ]
            },
            'sweet_pink': {
                'name': '甜美粉色妝',
                'effects': [
                    ('eyeshadow_pink', 'left_eye', 0.5),
                    ('eyeshadow_pink', 'right_eye', 0.5),
                    ('eyebrow_brown', 'left_eyebrow', 0.4),
                    ('eyebrow_brown', 'right_eyebrow', 0.4),
                    ('lipstick_pink', 'lips', 0.6),
                    ('blush_pink', 'left_cheek', 0.4),
                    ('blush_pink', 'right_cheek', 0.4),
                    ('highlight_champagne', 'forehead', 0.2),
                ]
            },
            'korean_style': {
                'name': '韓系清透妝',
                'effects': [
                    ('eyeshadow_brown', 'left_eye_upper', 0.2),
                    ('eyeshadow_brown', 'right_eye_upper', 0.2),
                    ('eyebrow_brown', 'left_eyebrow', 0.3),
                    ('eyebrow_brown', 'right_eyebrow', 0.3),
                    ('lipstick_coral', 'lips', 0.4),
                    ('blush_peach', 'left_cheek', 0.3),
                    ('blush_peach', 'right_cheek', 0.3),
                ]
            },
            'full_glam': {
                'name': '完整魅力妝',
                'effects': [
                    ('eyeshadow_gold', 'left_eye', 0.5),
                    ('eyeshadow_gold', 'right_eye', 0.5),
                    ('eyeshadow_brown', 'left_eye_lower', 0.3),
                    ('eyeshadow_brown', 'right_eye_lower', 0.3),
                    ('eyeliner_black', 'left_eye_upper', 0.6),
                    ('eyeliner_black', 'right_eye_upper', 0.6),
                    ('eyebrow_black', 'left_eyebrow', 0.5),
                    ('eyebrow_black', 'right_eyebrow', 0.5),
                    ('lipstick_berry', 'lips', 0.7),
                    ('blush_coral', 'left_cheek', 0.4),
                    ('blush_coral', 'right_cheek', 0.4),
                    ('highlight_gold', 'forehead', 0.3),
                    ('contour_brown', 'nose', 0.4),
                    ('nose_contour', 'nose_bridge', 0.3),
                ]
            },
            'professional_nude': {
                'name': '專業裸妝',
                'effects': [
                    ('eyeshadow_neutral', 'left_eye_upper', 0.2),
                    ('eyeshadow_neutral', 'right_eye_upper', 0.2),
                    ('eyebrow_taupe', 'left_eyebrow', 0.3),
                    ('eyebrow_taupe', 'right_eyebrow', 0.3),
                    ('lipstick_nude', 'lips', 0.4),
                    ('blush_peach', 'left_cheek', 0.2),
                    ('blush_peach', 'right_cheek', 0.2),
                    ('highlight_pearl', 'forehead', 0.15),
                    ('contour_taupe', 'nose_bridge', 0.2),
                ]
            },
            'vintage_red': {
                'name': '復古紅唇妝',
                'effects': [
                    ('eyeshadow_brown', 'left_eye', 0.3),
                    ('eyeshadow_brown', 'right_eye', 0.3),
                    ('eyeliner_black', 'left_eye_upper', 0.5),
                    ('eyeliner_black', 'right_eye_upper', 0.5),
                    ('eyebrow_black', 'left_eyebrow', 0.6),
                    ('eyebrow_black', 'right_eyebrow', 0.6),
                    ('lipstick_red', 'lips', 0.8),
                    ('blush_rose', 'left_cheek', 0.3),
                    ('blush_rose', 'right_cheek', 0.3),
                ]
            },
            'bronze_goddess': {
                'name': '青銅女神妝',
                'effects': [
                    ('eyeshadow_bronze', 'left_eye', 0.6),
                    ('eyeshadow_bronze', 'right_eye', 0.6),
                    ('eyeshadow_gold', 'left_eye_upper', 0.3),
                    ('eyeshadow_gold', 'right_eye_upper', 0.3),
                    ('eyeliner_brown', 'left_eye_upper', 0.4),
                    ('eyeliner_brown', 'right_eye_upper', 0.4),
                    ('eyebrow_brown', 'left_eyebrow', 0.5),
                    ('eyebrow_brown', 'right_eyebrow', 0.5),
                    ('lipstick_coral', 'lips', 0.6),
                    ('blush_coral', 'left_cheek', 0.4),
                    ('blush_coral', 'right_cheek', 0.4),
                    ('highlight_gold', 'forehead', 0.4),
                    ('contour_brown', 'nose', 0.3),
                ]
            },
            'wine_elegance': {
                'name': '酒紅優雅妝',
                'effects': [
                    ('eyeshadow_smoky', 'left_eye', 0.4),
                    ('eyeshadow_smoky', 'right_eye', 0.4),
                    ('eyeshadow_brown', 'left_eye_lower', 0.2),
                    ('eyeshadow_brown', 'right_eye_lower', 0.2),
                    ('eyeliner_black', 'left_eye_upper', 0.5),
                    ('eyeliner_black', 'right_eye_upper', 0.5),
                    ('eyebrow_black', 'left_eyebrow', 0.4),
                    ('eyebrow_black', 'right_eyebrow', 0.4),
                    ('lipstick_wine', 'lips', 0.7),
                    ('blush_rose', 'left_cheek', 0.3),
                    ('blush_rose', 'right_cheek', 0.3),
                    ('highlight_champagne', 'forehead', 0.2),
                ]
            }
        }
    
    def detect_face_landmarks(self, image: np.ndarray) -> Optional[List]:
        """檢測人臉關鍵點"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0]
            return None
    
    def create_region_mask(self, image_shape: Tuple[int, int], landmarks, region: str,
                          dilate_kernel: int = 8) -> np.ndarray:
        """創建指定區域的蒙版"""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if region not in self.FACE_REGIONS:
            print(f"⚠️  未知的面部區域: {region}")
            return mask
        
        # 提取關鍵點座標
        region_points = []
        for idx in self.FACE_REGIONS[region]:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                region_points.append([x, y])
        
        if len(region_points) < 3:
            print(f"⚠️  {region} 區域關鍵點不足")
            return mask
        
        # 創建多邊形蒙版
        points = np.array(region_points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        # 膨脹和模糊處理
        if dilate_kernel > 0:
            kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            # 高斯模糊使邊緣更自然
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
        
        return mask
    
    def apply_color_makeup(self, image: np.ndarray, mask: np.ndarray,
                          color: Tuple[int, int, int], alpha: float = 0.5,
                          blend_mode: str = 'normal') -> np.ndarray:
        """
        在指定區域應用顏色化妝
        
        Args:
            image: 原始圖片
            mask: 區域蒙版
            color: BGR 顏色
            alpha: 透明度 (0-1)
            blend_mode: 混合模式 ('normal', 'multiply', 'overlay')
            
        Returns:
            應用化妝後的圖片
        """
        result = image.copy().astype(np.float32)
        
        # 創建彩色覆蓋層
        colored_mask = np.zeros_like(image, dtype=np.float32)
        colored_mask[mask > 0] = color
        
        # 正規化蒙版
        normalized_mask = mask.astype(np.float32) / 255.0
        
        # 根據混合模式應用顏色
        if blend_mode == 'multiply':
            # 正片疊底模式 - 適合陰影和修容
            for c in range(3):
                result[:, :, c] = (
                    image[:, :, c] * (1 - normalized_mask * alpha) +
                    (image[:, :, c] * colored_mask[:, :, c] / 255.0) * normalized_mask * alpha
                )
        elif blend_mode == 'overlay':
            # 疊加模式 - 適合高光
            for c in range(3):
                base = image[:, :, c] / 255.0
                overlay = colored_mask[:, :, c] / 255.0
                
                blended = np.where(
                    base < 0.5,
                    2 * base * overlay,
                    1 - 2 * (1 - base) * (1 - overlay)
                )
                
                result[:, :, c] = (
                    image[:, :, c] * (1 - normalized_mask * alpha) +
                    blended * 255 * normalized_mask * alpha
                )
        else:  # normal
            # 正常混合模式
            for c in range(3):
                result[:, :, c] = (
                    image[:, :, c] * (1 - normalized_mask * alpha) +
                    colored_mask[:, :, c] * normalized_mask * alpha
                )
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_single_effect(self, image: np.ndarray, landmarks, 
                           color_name: str, region: str, alpha: float) -> np.ndarray:
        """應用單個化妝效果"""
        if color_name not in self.MAKEUP_COLORS:
            print(f"⚠️  未知的顏色: {color_name}")
            return image
        
        # 創建蒙版
        mask = self.create_region_mask(image.shape[:2], landmarks, region)
        
        if np.sum(mask) == 0:
            print(f"⚠️  {region} 區域蒙版為空")
            return image
        
        # 獲取顏色
        color = self.MAKEUP_COLORS[color_name]
        
        # 選擇混合模式
        blend_mode = 'normal'
        if 'contour' in color_name or 'eyeshadow_smoky' in color_name:
            blend_mode = 'multiply'
        elif 'highlight' in color_name:
            blend_mode = 'overlay'
        
        # 應用化妝
        return self.apply_color_makeup(image, mask, color, alpha, blend_mode)
    
    def apply_makeup_style(self, image_path: str, style_name: str,
                          output_path: str = None, show_process: bool = True) -> Optional[np.ndarray]:
        """
        應用指定的化妝風格
        
        Args:
            image_path: 輸入圖片路徑
            style_name: 化妝風格名稱
            output_path: 輸出路徑
            show_process: 是否顯示處理過程
            
        Returns:
            處理後的圖片
        """
        if style_name not in self.MAKEUP_STYLES:
            print(f"❌ 不支持的化妝風格: {style_name}")
            print(f"支持的風格: {list(self.MAKEUP_STYLES.keys())}")
            return None
        
        if not os.path.exists(image_path):
            print(f"❌ 圖片文件不存在: {image_path}")
            return None
        
        style = self.MAKEUP_STYLES[style_name]
        print(f"💄 開始應用 {style['name']} 化妝風格...")
        
        # 1. 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 無法讀取圖片: {image_path}")
            return None
        
        print(f"✅ 圖片載入成功 - 尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 2. 檢測人臉關鍵點
        landmarks = self.detect_face_landmarks(image)
        if landmarks is None:
            print("❌ 未檢測到人臉")
            return None
        
        print("✅ 人臉檢測成功")
        
        # 3. 依序應用所有化妝效果
        result = image.copy()
        
        for i, (color_name, region, alpha) in enumerate(style['effects']):
            print(f"  🎨 步驟 {i+1}/{len(style['effects'])}: 應用 {color_name} 到 {region} (透明度: {alpha})")
            result = self.apply_single_effect(result, landmarks, color_name, region, alpha)
        
        print(f"✅ {style['name']} 化妝完成！")
        
        # 4. 顯示處理過程
        if show_process:
            self._visualize_makeup_process(image, result, style['name'])
        
        # 5. 保存結果
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"💾 結果已保存: {output_path}")
        
        return result
    
    def apply_all_styles(self, image_path: str, show_comparison: bool = True) -> Dict[str, np.ndarray]:
        """應用所有化妝風格並比較效果"""
        if not os.path.exists(image_path):
            print(f"❌ 圖片文件不存在: {image_path}")
            return {}
        
        print("🎨 應用所有化妝風格...")
        
        results = {}
        original = cv2.imread(image_path)
        
        for style_name in self.MAKEUP_STYLES.keys():
            print(f"\n📋 處理風格: {self.MAKEUP_STYLES[style_name]['name']}")
            
            result = self.apply_makeup_style(
                image_path=image_path,
                style_name=style_name,
                output_path=f"full_makeup_{style_name}.jpg",
                show_process=False
            )
            
            if result is not None:
                results[style_name] = result
                print(f"✅ {style_name} 完成")
            else:
                print(f"❌ {style_name} 失敗")
        
        # 顯示所有風格的比較
        if show_comparison and results:
            self._show_all_styles_comparison(original, results)
        
        return results
    
    def _visualize_makeup_process(self, original: np.ndarray, result: np.ndarray, style_name: str):
        """可視化化妝過程"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 原始圖片
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=14)
        axes[0].axis('off')
        
        # 結果圖片
        axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Result: {style_name}", fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"makeup_process_{style_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def _show_all_styles_comparison(self, original: np.ndarray, results: Dict[str, np.ndarray]):
        """顯示所有風格的比較"""
        num_styles = len(results)
        cols = 3
        rows = (num_styles + 2) // cols  # +1 for original
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        # 原始圖片
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=12, weight='bold')
        axes[0].axis('off')
        
        # 各種風格的結果
        for i, (style_name, result) in enumerate(results.items(), 1):
            if i < len(axes):
                axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                axes[i].set_title(self.MAKEUP_STYLES[style_name]['name'], fontsize=12)
                axes[i].axis('off')
        
        # 隱藏多餘的子圖
        for i in range(len(results) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("all_makeup_styles_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """主函數演示"""
    print("💄 完整化妝應用程序演示")
    print("=" * 60)
    
    # 初始化應用程序
    app = FullMakeupApplication()
    
    # 測試圖片
    test_image = "my_face.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ 找不到測試圖片: {test_image}")
        print("請準備一張人臉圖片並命名為 'my_face.jpg'")
        return
    
    print("\n📋 可用的化妝風格:")
    for style_name, style_info in app.MAKEUP_STYLES.items():
        print(f"  - {style_name}: {style_info['name']}")
    
    # 選項1: 應用單個風格
    print("\n🎨 測試單個化妝風格...")
    
    # 測試自然日常妝
    result = app.apply_makeup_style(
        image_path=test_image,
        style_name='natural_daily',
        output_path="natural_daily_makeup.jpg",
        show_process=True
    )
    
    # 測試魅力晚妝
    result = app.apply_makeup_style(
        image_path=test_image,
        style_name='glamorous_evening',
        output_path="glamorous_evening_makeup.jpg", 
        show_process=True
    )
    
    # 選項2: 應用所有風格並比較
    print("\n🎨 應用所有化妝風格並比較...")
    
    all_results = app.apply_all_styles(
        image_path=test_image,
        show_comparison=True
    )
    
    print(f"\n✅ 完成！共生成 {len(all_results)} 種化妝風格")
    
    print("\n📝 生成的文件:")
    print("  - natural_daily_makeup.jpg")
    print("  - glamorous_evening_makeup.jpg")
    for style_name in all_results.keys():
        print(f"  - full_makeup_{style_name}.jpg")
    print("  - all_makeup_styles_comparison.png")
    
    print("\n🎉 完整化妝應用程序演示完成！")


if __name__ == "__main__":
    main()