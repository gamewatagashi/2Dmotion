import math
import random
from pathlib import Path

import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import moviepy.editor as mpy


# ==========================================
# 設定
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

AUDIO_PATH = BASE_DIR / "music.mp3"
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_PATH = BASE_DIR / "output_fast.mp4"

WIDTH = 1920
HEIGHT = 1080
FPS = 30

# パーティクル数
PARTICLE_COUNT = 200

# ==========================================
# ファイル確認
# ==========================================

if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"音楽ファイルが見つかりません:\n{AUDIO_PATH}"
    )

IMAGE_DIR.mkdir(exist_ok=True)

print("==========================================")
print(" 高速MV自動生成プログラム")
print("==========================================")
print()

# ==========================================
# 画像読み込み
# ==========================================

IMAGE_FILES = sorted(
    list(IMAGE_DIR.glob("*.jpg"))
    + list(IMAGE_DIR.glob("*.jpeg"))
    + list(IMAGE_DIR.glob("*.png"))
    + list(IMAGE_DIR.glob("*.webp"))
)

loaded_images = []

print(f"画像素材を読み込んでいます... {len(IMAGE_FILES)}枚")

for image_file in IMAGE_FILES:
    try:
        img = Image.open(image_file).convert("RGBA")
        loaded_images.append(img)
        print(f"  読み込み: {image_file.name}")
    except Exception as e:
        print(f"  スキップ: {image_file.name} ({e})")

print()

# ==========================================
# 音楽解析
# ==========================================

print("1/3: 音楽ファイルを解析中...")

y, sr = librosa.load(
    str(AUDIO_PATH),
    sr=None,
    mono=True
)

duration = librosa.get_duration(
    y=y,
    sr=sr
)

print(f"音源時間: {duration:.2f}秒")
print(f"サンプリングレート: {sr} Hz")

# ==========================================
# 音の強さ / Onset解析
# ==========================================

onset_env = librosa.onset.onset_strength(
    y=y,
    sr=sr
)

max_val = float(np.max(onset_env))

if max_val <= 0:
    max_val = 1.0


def get_intensity(t):
    """
    時刻tにおける音の強さを0～1で返す
    """

    if duration <= 0:
        return 0.0

    idx = int((t / duration) * len(onset_env))

    idx = min(
        max(idx, 0),
        len(onset_env) - 1
    )

    intensity = float(onset_env[idx]) / max_val

    return max(
        0.0,
        min(1.0, intensity)
    )


# ==========================================
# パーティクル初期化
# ==========================================

particles = []

for _ in range(PARTICLE_COUNT):
    particles.append(
        {
            "x": random.uniform(-1000, 1000),
            "y": random.uniform(-1000, 1000),
            "z": random.uniform(10, 1000),
        }
    )


# ==========================================
# 画面中央
# ==========================================

cx = WIDTH // 2
cy = HEIGHT // 2


# ==========================================
# フォント
# ==========================================

def get_font(size):
    """
    Windows / Linux / macOSで可能な限り
    利用できるフォントを探す。
    """

    candidates = [
        # Windows
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/meiryo.ttc"),

        # Linux
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),

        # macOS
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]

    for font_path in candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(
                    str(font_path),
                    size
                )
            except Exception:
                pass

    return ImageFont.load_default()


# ==========================================
# シーン管理
# ==========================================

def get_scene_id(t):
    """
    約0.67秒ごとにシーンを変更する。

    時刻を元にseedを作ることで、
    同じ時刻では同じシーンになる。
    """

    seed = int(t * 1.5)

    rng = random.Random(seed)

    return rng.randint(0, 3)


# ==========================================
# フレーム生成
# ==========================================

def make_frame(t):

    intensity = get_intensity(t)

    scene_id = get_scene_id(t)

    # --------------------------------------
    # フレームごとの乱数
    # --------------------------------------

    rng = random.Random(
        int(t * FPS)
    )

    # --------------------------------------
    # カメラシェイク
    # --------------------------------------

    if intensity > 0.4:

        shake_x = int(
            (rng.random() - 0.5)
            * intensity
            * 50
        )

        shake_y = int(
            (rng.random() - 0.5)
            * intensity
            * 50
        )

    else:

        shake_x = 0
        shake_y = 0

    # --------------------------------------
    # 背景
    # --------------------------------------

    base_img = Image.new(
        "RGBA",
        (WIDTH, HEIGHT),
        (10, 10, 15, 255)
    )

    draw = ImageDraw.Draw(base_img)

    # ======================================
    # SCENE 0
    # 3Dスピードワープ
    # ======================================

    if scene_id == 0:

        for p in particles:

            p["z"] -= (
                20
                + intensity * 80
            )

            if p["z"] <= 10:

                p["z"] = 1000

                p["x"] = rng.uniform(
                    -1000,
                    1000
                )

                p["y"] = rng.uniform(
                    -1000,
                    1000
                )

            px = int(
                cx
                + shake_x
                + (p["x"] * 400)
                / p["z"]
            )

            py = int(
                cy
                + shake_y
                + (p["y"] * 400)
                / p["z"]
            )

            size = max(
                1,
                int(
                    (1000 - p["z"])
                    / 150
                )
            )

            if (
                0 <= px < WIDTH
                and
                0 <= py < HEIGHT
            ):

                draw.ellipse(
                    [
                        px - size,
                        py - size,
                        px + size,
                        py + size
                    ],
                    fill=(0, 255, 200, 255)
                )

        # ----------------------------------
        # 集中線
        # ----------------------------------

        num_lines = int(
            20 + intensity * 40
        )

        for _ in range(num_lines):

            angle = rng.uniform(
                0,
                math.pi * 2
            )

            r1 = rng.uniform(
                200,
                400
            )

            r2 = (
                r1
                + rng.uniform(
                    200,
                    600
                )
            )

            x1 = (
                cx
                + math.cos(angle) * r1
            )

            y1 = (
                cy
                + math.sin(angle) * r1
            )

            x2 = (
                cx
                + math.cos(angle) * r2
            )

            y2 = (
                cy
                + math.sin(angle) * r2
            )

            draw.line(
                [(x1, y1), (x2, y2)],
                fill=(255, 255, 255, 180),
                width=rng.randint(1, 4)
            )

    # ======================================
    # SCENE 1
    # 画像高速カットイン
    # ======================================

    elif scene_id == 1 and loaded_images:

        img_idx = (
            int(t * 15)
            % len(loaded_images)
        )

        photo = loaded_images[
            img_idx
        ].copy()

        # ----------------------------------
        # 音に合わせたズーム
        # ----------------------------------

        scale = (
            0.5
            + intensity * 0.3
        )

        nw = max(
            1,
            int(photo.width * scale)
        )

        nh = max(
            1,
            int(photo.height * scale)
        )

        photo = photo.resize(
            (nw, nh),
            Image.Resampling.LANCZOS
        )

        px = (
            cx
            - nw // 2
            + shake_x
        )

        py = (
            cy
            - nh // 2
            + shake_y
        )

        base_img.paste(
            photo,
            (px, py),
            photo
        )

    # ======================================
    # SCENE 2
    # 回転ワイヤーフレーム
    # ======================================

    elif scene_id == 2:

        angle = t * 6.0

        scale = (
            150
            + intensity * 200
        )

        for i in range(8):

            a = (
                angle
                + i * math.pi / 4
            )

            x1 = int(
                cx
                + shake_x
                + math.cos(a)
                * scale
            )

            y1 = int(
                cy
                + shake_y
                + math.sin(a)
                * scale
            )

            x2 = int(
                cx
                + shake_x
                + math.cos(a + 0.5)
                * (scale * 0.5)
            )

            y2 = int(
                cy
                + shake_y
                + math.sin(a + 0.5)
                * (scale * 0.5)
            )

            draw.line(
                [
                    (x1, y1),
                    (x2, y2)
                ],
                fill=(255, 0, 110, 255),
                width=5
            )

    # ======================================
    # SCENE 3
    # グリッチタイポグラフィ
    # ======================================

    else:

        font = get_font(
            int(
                100
                + intensity * 50
            )
        )

        if int(t * 8) % 2 == 0:
            txt = "SPEED"
        else:
            txt = "FAST"

        # ----------------------------------
        # グリッチっぽい影
        # ----------------------------------

        draw.text(
            (
                cx - 150 + shake_x + 8,
                cy - 50 + shake_y
            ),
            txt,
            fill=(255, 0, 80, 180),
            font=font
        )

        draw.text(
            (
                cx - 150 + shake_x - 8,
                cy - 50 + shake_y
            ),
            txt,
            fill=(0, 255, 255, 180),
            font=font
        )

        draw.text(
            (
                cx - 150 + shake_x,
                cy - 50 + shake_y
            ),
            txt,
            fill=(255, 255, 255, 255),
            font=font
        )

    # ==========================================
    # 全シーン共通エフェクト
    # ==========================================

    # ------------------------------------------
    # 水平スリットノイズ
    # ------------------------------------------

    if intensity > 0.5:

        count = int(
            intensity * 12
        )

        for _ in range(count):

            ny = rng.randint(
                0,
                HEIGHT - 1
            )

            nh = rng.randint(
                3,
                12
            )

            draw.rectangle(
                [
                    0,
                    ny,
                    WIDTH,
                    min(
                        HEIGHT,
                        ny + nh
                    )
                ],
                fill=(255, 255, 255, 200)
            )

    # ------------------------------------------
    # RGB変換
    # ------------------------------------------

    final_frame = base_img.convert(
        "RGB"
    )

    # ------------------------------------------
    # 強ビート時のネガポジ反転
    # ------------------------------------------

    if intensity > 0.8:

        final_frame = ImageOps.invert(
            final_frame
        )

    return np.array(
        final_frame
    )


# ==========================================
# 動画生成
# ==========================================

print()
print("2/3: 疾走感のあるMVを生成中...")
print()
print("※音源が長い場合はかなり時間がかかります。")
print()

video_clip = mpy.VideoClip(
    make_frame,
    duration=duration
)

audio_clip = mpy.AudioFileClip(
    str(AUDIO_PATH)
)

final_clip = video_clip.set_audio(
    audio_clip
)

# ==========================================
# 出力
# ==========================================

print()
print("3/3: MP4を書き出しています...")
print()

final_clip.write_videofile(
    str(OUTPUT_PATH),
    fps=FPS,
    codec="libx264",
    audio_codec="aac",
    preset="medium",
    threads=4
)

# ==========================================
# 後処理
# ==========================================

video_clip.close()
audio_clip.close()
final_clip.close()

print()
print("==========================================")
print("完成！")
print(f"出力ファイル: {OUTPUT_PATH}")
print("==========================================")
