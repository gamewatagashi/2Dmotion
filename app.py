import io
import math
import random
import tempfile
from pathlib import Path

import librosa
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import moviepy.editor as mpy


# ============================================================
# ページ設定
# ============================================================

st.set_page_config(
    page_title="MV Generator",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# タイトル
# ============================================================

st.title("🎬 MV Generator")
st.caption(
    "音楽を解析して、ビート・盛り上がり・音量に合わせて"
    "映像を自動生成します。"
)


# ============================================================
# サイドバー設定
# ============================================================

st.sidebar.header("⚙️ MV設定")

mood = st.sidebar.selectbox(
    "🎨 映像の雰囲気",
    [
        "おまかせ",
        "疾走感",
        "激しい",
        "ダーク",
        "サイバー",
        "レトロ",
        "幻想的",
        "明るい",
    ]
)

intensity_setting = st.sidebar.slider(
    "⚡ 演出の激しさ",
    min_value=1,
    max_value=10,
    value=7
)

randomness = st.sidebar.slider(
    "🎲 ランダム性",
    min_value=0,
    max_value=10,
    value=7
)

image_speed = st.sidebar.select_slider(
    "🖼️ 画像切り替え速度",
    options=[
        "ゆっくり",
        "標準",
        "高速",
        "超高速"
    ],
    value="高速"
)

aspect_ratio = st.sidebar.selectbox(
    "📐 アスペクト比",
    [
        "16:9",
        "9:16"
    ]
)

camera_shake = st.sidebar.checkbox(
    "📳 カメラシェイク",
    value=True
)

glitch = st.sidebar.checkbox(
    "💥 グリッチ",
    value=True
)

flash = st.sidebar.checkbox(
    "✨ フラッシュ",
    value=True
)

invert = st.sidebar.checkbox(
    "🔄 強ビート時の色反転",
    value=False
)


# ============================================================
# アップロード
# ============================================================

st.header("① 素材")

audio_file = st.file_uploader(
    "🎵 音楽ファイル",
    type=[
        "mp3",
        "wav",
        "m4a",
        "ogg"
    ]
)

image_files = st.file_uploader(
    "🖼️ 写真・画像",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    accept_multiple_files=True
)


# ============================================================
# 画像読み込み
# ============================================================

loaded_images = []

if image_files:

    for file in image_files:

        try:

            image = Image.open(file).convert("RGBA")

            loaded_images.append(image)

        except Exception as e:

            st.warning(
                f"{file.name} を読み込めませんでした: {e}"
            )


if loaded_images:

    st.write(
        f"📷 {len(loaded_images)}枚の画像を読み込みました。"
    )

    cols = st.columns(
        min(5, len(loaded_images))
    )

    for i, image in enumerate(loaded_images[:5]):

        cols[i].image(
            image,
            use_container_width=True
        )

    if len(loaded_images) > 5:

        st.caption(
            f"ほか {len(loaded_images) - 5}枚"
        )


# ============================================================
# アスペクト比
# ============================================================

if aspect_ratio == "16:9":

    WIDTH = 1280
    HEIGHT = 720

else:

    WIDTH = 720
    HEIGHT = 1280


# ============================================================
# フォント
# ============================================================

def get_font(size):

    candidates = [

        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/meiryo.ttc",

        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",

        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    for path in candidates:

        if Path(path).exists():

            try:

                return ImageFont.truetype(
                    path,
                    size
                )

            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# 画像フィット
# ============================================================

def fit_image(image, width, height):

    image = image.copy()

    target_ratio = width / height
    image_ratio = image.width / image.height

    if image_ratio > target_ratio:

        new_height = height

        new_width = int(
            height * image_ratio
        )

    else:

        new_width = width

        new_height = int(
            width / image_ratio
        )

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    left = (
        new_width - width
    ) // 2

    top = (
        new_height - height
    ) // 2

    image = image.crop(
        (
            left,
            top,
            left + width,
            top + height
        )
    )

    return image


# ============================================================
# 雰囲気処理
# ============================================================

def apply_mood(image, selected_mood):

    image = image.convert("RGBA")

    if selected_mood == "ダーク":

        image = ImageEnhance.Brightness(
            image
        ).enhance(0.55)

        image = ImageEnhance.Contrast(
            image
        ).enhance(1.4)

    elif selected_mood == "明るい":

        image = ImageEnhance.Brightness(
            image
        ).enhance(1.25)

        image = ImageEnhance.Color(
            image
        ).enhance(1.3)

    elif selected_mood == "レトロ":

        image = ImageEnhance.Color(
            image
        ).enhance(0.65)

        image = ImageEnhance.Contrast(
            image
        ).enhance(1.15)

    elif selected_mood == "サイバー":

        image = ImageEnhance.Contrast(
            image
        ).enhance(1.5)

        image = ImageEnhance.Color(
            image
        ).enhance(1.5)

    elif selected_mood == "幻想的":

        image = ImageEnhance.Brightness(
            image
        ).enhance(1.1)

        image = ImageEnhance.Color(
            image
        ).enhance(0.8)

    elif selected_mood == "激しい":

        image = ImageEnhance.Contrast(
            image
        ).enhance(1.7)

        image = ImageEnhance.Color(
            image
        ).enhance(1.4)

    elif selected_mood == "疾走感":

        image = ImageEnhance.Contrast(
            image
        ).enhance(1.4)

        image = ImageEnhance.Color(
            image
        ).enhance(1.2)

    return image


# ============================================================
# 音楽解析
# ============================================================

@st.cache_data
def analyze_audio(audio_bytes):

    with tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    ) as temp:

        temp.write(audio_bytes)

        audio_path = temp.name

    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    # ----------------------------
    # Onset
    # ----------------------------

    onset = librosa.onset.onset_strength(
        y=y,
        sr=sr
    )

    if len(onset) == 0:

        onset = np.array([0])

    max_onset = np.max(onset)

    if max_onset > 0:

        normalized_onset = (
            onset / max_onset
        )

    else:

        normalized_onset = onset

    # ----------------------------
    # BPM
    # ----------------------------

    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    try:

        bpm = float(np.asarray(tempo).flatten()[0])

    except Exception:

        bpm = 120.0

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr
    )

    # ----------------------------
    # RMS
    # ----------------------------

    rms = librosa.feature.rms(
        y=y
    )[0]

    max_rms = np.max(rms)

    if max_rms > 0:

        normalized_rms = (
            rms / max_rms
        )

    else:

        normalized_rms = rms

    return {
        "duration": duration,
        "sr": sr,
        "onset": normalized_onset,
        "rms": normalized_rms,
        "bpm": bpm,
        "beat_times": beat_times,
    }


# ============================================================
# 音量取得
# ============================================================

def get_rms(
    t,
    rms,
    duration
):

    if len(rms) == 0:

        return 0.0

    index = int(
        (t / duration)
        * len(rms)
    )

    index = max(
        0,
        min(
            index,
            len(rms) - 1
        )
    )

    return float(
        rms[index]
    )


# ============================================================
# Onset取得
# ============================================================

def get_onset(
    t,
    onset,
    duration
):

    if len(onset) == 0:

        return 0.0

    index = int(
        (t / duration)
        * len(onset)
    )

    index = max(
        0,
        min(
            index,
            len(onset) - 1
        )
    )

    return float(
        onset[index]
    )


# ============================================================
# 演出決定
# ============================================================

def choose_effect(
    intensity,
    rms,
    mood,
    rng
):

    effects = [
        "zoom",
        "shake",
        "flash",
        "glitch",
        "rotate",
        "speedline",
        "normal",
    ]

    # ----------------------------------------
    # 音が弱い
    # ----------------------------------------

    if intensity < 0.25:

        effects = [
            "zoom",
            "normal",
            "zoom"
        ]

    # ----------------------------------------
    # 音が普通
    # ----------------------------------------

    elif intensity < 0.55:

        effects = [
            "zoom",
            "rotate",
            "normal",
            "glitch"
        ]

    # ----------------------------------------
    # 強いビート
    # ----------------------------------------

    elif intensity < 0.8:

        effects = [
            "shake",
            "zoom",
            "glitch",
            "flash",
            "rotate"
        ]

    # ----------------------------------------
    # 超強ビート
    # ----------------------------------------

    else:

        effects = [
            "shake",
            "flash",
            "glitch",
            "speedline",
            "rotate"
        ]

    # ----------------------------------------
    # 雰囲気補正
    # ----------------------------------------

    if mood == "疾走感":

        effects += [
            "speedline",
            "zoom",
            "shake"
        ]

    elif mood == "激しい":

        effects += [
            "glitch",
            "flash",
            "shake"
        ]

    elif mood == "サイバー":

        effects += [
            "glitch",
            "rotate"
        ]

    elif mood == "ダーク":

        effects += [
            "zoom",
            "glitch"
        ]

    # ----------------------------------------
    # ランダム選択
    # ----------------------------------------

    return rng.choice(
        effects
    )


# ============================================================
# MV生成
# ============================================================

def generate_mv(
    audio_bytes,
    images,
    analysis,
    settings,
    progress_callback=None
):

    duration = analysis["duration"]

    onset = analysis["onset"]

    rms = analysis["rms"]

    bpm = analysis["bpm"]

    mood = settings["mood"]

    randomness = settings["randomness"]

    intensity_setting = (
        settings["intensity"]
    )

    fps = 24

    # ----------------------------------------
    # ランダムシード
    # ----------------------------------------

    seed = random.randint(
        0,
        999999999
    )

    rng = random.Random(seed)

    # ----------------------------------------
    # 画像がない場合
    # ----------------------------------------

    if not images:

        images = []

    # ----------------------------------------
    # 一時ファイル
    # ----------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    ) as temp:

        temp.write(audio_bytes)

        audio_path = temp.name

    # ----------------------------------------
    # 音声
    # ----------------------------------------

    audio_clip = mpy.AudioFileClip(
        audio_path
    )

    # ----------------------------------------
    # 画像切り替え間隔
    # ----------------------------------------

    speed_map = {
        "ゆっくり": 2.5,
        "標準": 1.2,
        "高速": 0.6,
        "超高速": 0.25
    }

    base_interval = speed_map[
        settings["image_speed"]
    ]

    # ----------------------------------------
    # シーン用状態
    # ----------------------------------------

    scene_seed = rng.randint(
        0,
        999999
    )

    def make_frame(t):

        nonlocal scene_seed

        local_rng = random.Random(
            scene_seed
            + int(t * fps)
        )

        current_onset = get_onset(
            t,
            onset,
            duration
        )

        current_rms = get_rms(
            t,
            rms,
            duration
        )

        # ------------------------------------
        # 音楽の強度
        # ------------------------------------

        intensity = (
            current_onset * 0.65
            + current_rms * 0.35
        )

        intensity *= (
            intensity_setting / 7
        )

        intensity = max(
            0,
            min(
                1,
                intensity
            )
        )

        # ------------------------------------
        # カメラシェイク
        # ------------------------------------

        shake_x = 0
        shake_y = 0

        if (
            settings["camera_shake"]
            and intensity > 0.55
        ):

            amount = (
                intensity
                * 35
            )

            shake_x = int(
                local_rng.uniform(
                    -amount,
                    amount
                )
            )

            shake_y = int(
                local_rng.uniform(
                    -amount,
                    amount
                )
            )

        # ------------------------------------
        # 背景
        # ------------------------------------

        base = Image.new(
            "RGBA",
            (WIDTH, HEIGHT),
            (8, 8, 12, 255)
        )

        draw = ImageDraw.Draw(
            base
        )

        # ------------------------------------
        # 画像
        # ------------------------------------

        if images:

            image_index = int(
                t / base_interval
            ) % len(images)

            # ランダム性
            if randomness > 0:

                if local_rng.random() < (
                    randomness / 30
                ):

                    image_index = (
                        image_index
                        + local_rng.randint(
                            -2,
                            2
                        )
                    ) % len(images)

            image = images[
                image_index
            ].copy()

            image = fit_image(
                image,
                WIDTH,
                HEIGHT
            )

            image = apply_mood(
                image,
                mood
            )

            # --------------------------------
            # ズーム
            # --------------------------------

            zoom = (
                1
                + intensity * 0.12
            )

            if (
                choose_effect(
                    intensity,
                    current_rms,
                    mood,
                    local_rng
                )
                == "zoom"
            ):

                zoom += 0.08

            nw = int(
                WIDTH * zoom
            )

            nh = int(
                HEIGHT * zoom
            )

            image = image.resize(
                (nw, nh),
                Image.Resampling.LANCZOS
            )

            crop_x = (
                nw - WIDTH
            ) // 2

            crop_y = (
                nh - HEIGHT
            ) // 2

            image = image.crop(
                (
                    crop_x,
                    crop_y,
                    crop_x + WIDTH,
                    crop_y + HEIGHT
                )
            )

            base.alpha_composite(
                image,
                (
                    shake_x,
                    shake_y
                )
            )

        # ------------------------------------
        # 画像なし → 背景演出
        # ------------------------------------

        else:

            # パーティクル
            count = int(
                30
                + intensity * 120
            )

            for _ in range(count):

                x = local_rng.randint(
                    0,
                    WIDTH
                )

                y = local_rng.randint(
                    0,
                    HEIGHT
                )

                size = local_rng.randint(
                    1,
                    5
                )

                draw.ellipse(
                    [
                        x - size,
                        y - size,
                        x + size,
                        y + size
                    ],
                    fill=(
                        0,
                        255,
                        220,
                        180
                    )
                )

        # ====================================
        # エフェクト
        # ====================================

        effect = choose_effect(
            intensity,
            current_rms,
            mood,
            local_rng
        )

        # ------------------------------------
        # スピードライン
        # ------------------------------------

        if effect == "speedline":

            count = int(
                20
                + intensity * 70
            )

            for _ in range(count):

                angle = local_rng.uniform(
                    0,
                    math.pi * 2
                )

                r1 = local_rng.uniform(
                    150,
                    400
                )

                r2 = r1 + local_rng.uniform(
                    100,
                    700
                )

                x1 = (
                    WIDTH / 2
                    + math.cos(angle) * r1
                )

                y1 = (
                    HEIGHT / 2
                    + math.sin(angle) * r1
                )

                x2 = (
                    WIDTH / 2
                    + math.cos(angle) * r2
                )

                y2 = (
                    HEIGHT / 2
                    + math.sin(angle) * r2
                )

                draw.line(
                    [
                        (x1, y1),
                        (x2, y2)
                    ],
                    fill=(
                        255,
                        255,
                        255,
                        150
                    ),
                    width=local_rng.randint(
                        1,
                        5
                    )
                )

        # ------------------------------------
        # グリッチ
        # ------------------------------------

        if (
            settings["glitch"]
            and (
                effect == "glitch"
                or intensity > 0.75
            )
        ):

            for _ in range(
                int(
                    3
                    + intensity * 15
                )
            ):

                y = local_rng.randint(
                    0,
                    HEIGHT - 1
                )

                h = local_rng.randint(
                    2,
                    15
                )

                draw.rectangle(
                    [
                        0,
                        y,
                        WIDTH,
                        min(
                            HEIGHT - 1,
                            y + h
                        )
                    ],
                    fill=(
                        255,
                        255,
                        255,
                        100
                    )
                )

        # ------------------------------------
        # フラッシュ
        # ------------------------------------

        if (
            settings["flash"]
            and effect == "flash"
            and intensity > 0.7
        ):

            overlay = Image.new(
                "RGBA",
                (WIDTH, HEIGHT),
                (255, 255, 255, 150)
            )

            base = Image.alpha_composite(
                base,
                overlay
            )

        # ------------------------------------
        # 強ビート色反転
        # ------------------------------------

        final = base.convert(
            "RGB"
        )

        if (
            settings["invert"]
            and intensity > 0.9
        ):

            final = ImageOps.invert(
                final
            )

        return np.array(
            final
        )

    # ----------------------------------------
    # MoviePy
    # ----------------------------------------

    video = mpy.VideoClip(
        make_frame,
        duration=duration
    )

    video = video.set_audio(
        audio_clip
    )

    # ----------------------------------------
    # 出力
    # ----------------------------------------

    output_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )

    output_path = output_file.name

    output_file.close()

    video.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger=None
    )

    video.close()
    audio_clip.close()

    return output_path


# ============================================================
# メイン
# ============================================================

st.header("② 自動生成")

if audio_file:

    st.success(
        f"🎵 {audio_file.name}"
    )

    if st.button(
        "🎬 MVを自動生成する",
        type="primary",
        use_container_width=True
    ):

        audio_bytes = audio_file.getvalue()

        # ------------------------------------
        # 音楽解析
        # ------------------------------------

        with st.status(
            "🎵 音楽を解析しています...",
            expanded=True
        ) as status:

            analysis = analyze_audio(
                audio_bytes
            )

            st.write(
                f"⏱️ 曲の長さ: "
                f"{analysis['duration']:.1f}秒"
            )

            st.write(
                f"🥁 推定BPM: "
                f"{analysis['bpm']:.1f}"
            )

            st.write(
                "🎧 ビート・音量・音の強さを解析しました。"
            )

            status.update(
                label="音楽解析完了",
                state="complete"
            )

        # ------------------------------------
        # 設定
        # ------------------------------------

        settings = {
            "mood": mood,
            "intensity": intensity_setting,
            "randomness": randomness,
            "image_speed": image_speed,
            "camera_shake": camera_shake,
            "glitch": glitch,
            "flash": flash,
            "invert": invert
        }

        # ------------------------------------
        # MV生成
        # ------------------------------------

        progress = st.progress(
            0
        )

        status_text = st.empty()

        status_text.write(
            "🎬 MVを生成しています..."
        )

        try:

            output_path = generate_mv(
                audio_bytes,
                loaded_images,
                analysis,
                settings
            )

            progress.progress(
                100
            )

            status_text.write(
                "✅ MV生成完了！"
            )

            st.success(
                "🎉 MVが完成しました！"
            )

            # --------------------------------
            # 動画
            # --------------------------------

            with open(
                output_path,
                "rb"
            ) as f:

                video_bytes = f.read()

            st.video(
                video_bytes
            )

            # --------------------------------
            # ダウンロード
            # --------------------------------

            st.download_button(
                label="⬇️ MP4をダウンロード",
                data=video_bytes,
                file_name="generated_mv.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "MV生成中にエラーが発生しました。"
            )

            st.exception(e)

else:

    st.info(
        "まず音楽ファイルをアップロードしてください。"
    )


# ============================================================
# フッター
# ============================================================

st.divider()

st.caption(
    "MV Generator | Librosa + MoviePy + Streamlit"
)
