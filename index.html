import librosa
import numpy as np
import moviepy.editor as mpy
from PIL import Image, ImageDraw, ImageFont, ImageOps
import math
import random
import glob

# ==========================================
# 1. 設定パラメータ
# ==========================================
AUDIO_PATH = "music.mp3"         # 音楽ファイル
OUTPUT_PATH = "output_fast.mp4"  # 出力ファイル
WIDTH, HEIGHT = 1920, 1080       # フルHD

# 画像素材の読み込み（フォルダ内のjpg/pngを全取得）
IMAGE_FILES = glob.glob("*.jpg") + glob.glob("*.png")
loaded_images = []
for f in IMAGE_FILES:
    try:
        img = Image.open(f).convert("RGBA")
        loaded_images.append(img)
    except:
        pass

print("1/3: 音楽ファイルを詳細解析中...")
y, sr = librosa.load(AUDIO_PATH)
duration = librosa.get_duration(y=y, sr=sr)

# 音の立ち上がり（Onset / ビートのアタック感）を抽出
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
max_val = np.max(onset_env) if np.max(onset_env) > 0 else 1.0

def get_intensity(t):
    idx = int((t / duration) * len(onset_env))
    idx = min(max(idx, 0), len(onset_env) - 1)
    return onset_env[idx] / max_val

# 3Dパーティクル初期化
particles = [{'x': random.uniform(-1000, 1000), 'y': random.uniform(-1000, 1000), 
              'z': random.uniform(10, 1000)} for _ in range(200)]

print("2/3: 疾走感＆激しい場面展開のレンダリング開始...")
cx, cy = WIDTH // 2, HEIGHT // 2

# シーンのランダム管理用（2秒ごと、またはビートの切り替わりでチェンジ）
def get_scene_id(t):
    random.seed(int(t * 1.5)) # 1.5秒周期でシード変更（シーン切り替え）
    return random.randint(0, 3)

def make_frame(t):
    intensity = get_intensity(t)
    scene_id = get_scene_id(t)
    
    # --------------------------------------------------
    # 【激しい演出】カメラシェイク（画面揺れ）オフセット計算
    # --------------------------------------------------
    shake_x = int((random.random() - 0.5) * intensity * 50) if intensity > 0.4 else 0
    shake_y = int((random.random() - 0.5) * intensity * 50) if intensity > 0.4 else 0
    
    # ベースキャンバス作成
    base_img = Image.new("RGBA", (WIDTH, HEIGHT), (10, 10, 15, 255))
    draw = ImageDraw.Draw(base_img)
    
    # --------------------------------------------------
    # シーン分岐（場面展開）
    # --------------------------------------------------
    if scene_id == 0:
        # --- シーン0: 3Dスピードワープ＆集中線 ---
        for p in particles:
            p['z'] -= (20 + intensity * 80) # 高速移動
            if p['z'] <= 10:
                p['z'] = 1000
                p['x'], p['y'] = random.uniform(-1000, 1000), random.uniform(-1000, 1000)
            
            px = int(cx + shake_x + (p['x'] * 400) / p['z'])
            py = int(cy + shake_y + (p['y'] * 400) / p['z'])
            size = max(1, int((1000 - p['z']) / 150))
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                draw.ellipse([px-size, py-size, px+size, py+size], fill=(0, 255, 200))
                
        # 放射状スピードライン（集中線）
        num_lines = int(20 + intensity * 40)
        for _ in range(num_lines):
            angle = random.uniform(0, math.pi * 2)
            r1 = random.uniform(200, 400)
            r2 = r1 + random.uniform(200, 600)
            draw.line([(cx + math.cos(angle)*r1, cy + math.sin(angle)*r1),
                       (cx + math.cos(angle)*r2, cy + math.sin(angle)*r2)], 
                      fill=(255, 255, 255, 180), width=random.randint(1, 4))

    elif scene_id == 1 and loaded_images:
        # --- シーン1: MAD風・画像高速超コマ送りカットイン ---
        img_idx = int(t * 15) % len(loaded_images) # 毎秒15コマで爆速切り替え
        photo = loaded_images[img_idx].copy()
        
        # 音でズーム
        scale = 0.5 + intensity * 0.3
        nw, nh = int(photo.width * scale), int(photo.height * scale)
        if nw > 0 and nh > 0:
            photo = photo.resize((nw, nh), Image.Resampling.LANCZOS)
            px = cx - nw // 2 + shake_x
            py = cy - nh // 2 + shake_y
            base_img.paste(photo, (px, py), photo)

    elif scene_id == 2:
        # --- シーン2: 高速回転3Dワイヤーフレーム ＋ タイポグラフィ ---
        angle = t * 6.0 # 超高速回転
        scale = 150 + intensity * 200
        
        # キューブの回転描画
        for i in range(8):
            a = angle + (i * math.pi / 4)
            x1 = int(cx + shake_x + math.cos(a) * scale)
            y1 = int(cy + shake_y + math.sin(a) * scale)
            x2 = int(cx + shake_x + math.cos(a + 0.5) * (scale * 0.5))
            y2 = int(cy + shake_y + math.sin(a + 0.5) * (scale * 0.5))
            draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 110), width=5)

    else:
        # --- シーン3: 巨大グリッチタイポグラフィ ＋ 画面スリット ---
        try:
            font = ImageFont.truetype("arial.ttf", int(100 + intensity * 50))
        except:
            font = ImageFont.load_default()
            
        txt = "SPEED" if int(t * 8) % 2 == 0 else "FAST"
        draw.text((cx - 150 + shake_x, cy - 50 + shake_y), txt, fill=(255, 255, 255), font=font)

    # --------------------------------------------------
    # 【激しい演出】全シーン共通のエフェクト（ノイズ・フラッシュ）
    # --------------------------------------------------
    # 1. 画面全域のランダム水平スリットノイズ（ビート時）
    if intensity > 0.5:
        for _ in range(int(intensity * 12)):
            ny = random.randint(0, HEIGHT)
            nh = random.randint(3, 12)
            draw.rectangle([0, ny, WIDTH, ny + nh], fill=(255, 255, 255, 200))
            
    # RGBAからRGBに変換
    final_frame = base_img.convert("RGB")
    
    # 2. 【超強ビート時の色反転（ネガポジフラッシュ）】
    if intensity > 0.8:
        # ピーク音の瞬間に一瞬画面の色を完全反転
        final_frame = ImageOps.invert(final_frame)

    return np.array(final_frame)

# 動画出力
video_clip = mpy.VideoClip(make_frame, duration=duration)
audio_clip = mpy.AudioFileClip(AUDIO_PATH)
final_clip = video_clip.set_audio(audio_clip)

final_clip.write_videofile(OUTPUT_PATH, fps=30, codec="libx264", audio_codec="aac")
print("\n3/3: 完成！ output_fast.mp4 を再生してみてください。")
