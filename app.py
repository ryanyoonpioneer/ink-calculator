import streamlit as st
import numpy as np
import cv2
import pandas as pd
from skimage.color import rgb2lab, deltaE_ciede2000

st.set_page_config(page_title="Ink Coverage Calculator", layout="centered")
st.title("🎨 Ink Coverage Calculator")

MAX_PIXELS = 300000
DELTA_E_TOLERANCE = 18

st.info(
    "Gray background is treated as transparent / no ink. "
    "Tick White only if there is real white ink."
)

color_presets = {
    "Blue": "#1f2f8f",
    "Green": "#007a5a",
    "Red": "#d71920",
    "White": "#ffffff",
    "Black": "#000000",
    "Yellow": "#ffd200",
    "Orange": "#f58220",
    "Purple": "#6a1b9a",
    "Pink": "#e91e63",
    "Cyan / Blue-Green": "#00a6b4",
}

st.subheader("1️⃣ Tick actual printed ink colors")

selected_colors = {}
cols = st.columns(3)

for i, (color_name, default_hex) in enumerate(color_presets.items()):
    with cols[i % 3]:
        checked = st.checkbox(
            color_name,
            value=False,
            key=f"check_{color_name}",
        )

        if checked:
            selected_colors[color_name] = st.color_picker(
                f"{color_name} sample",
                default_hex,
                key=f"picker_{color_name}",
            )

if not selected_colors:
    st.warning("Please tick at least one actual ink color.")
    st.stop()

uploaded = st.file_uploader("📤 Upload artwork image", type=["png", "jpg", "jpeg"])

if not uploaded:
    st.info("Upload artwork after selecting the ink colors.")
    st.stop()

img_bytes = uploaded.read()
img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("Could not read image.")
    st.stop()

img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

st.subheader("2️⃣ Original image")
st.image(img_rgb_original, use_container_width=True)

if not st.button("Calculate Ink Coverage"):
    st.info("Click Calculate after uploading the image.")
    st.stop()


def hex_to_rgb_float(hex_color):
    hex_color = hex_color.replace("#", "")
    rgb = np.array(
        [
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        ],
        dtype=np.float32,
    )
    return rgb / 255.0


def make_transparent_background_mask(img_rgb):
    """
    Treat gray/light gray packaging preview background as transparent/no ink.
    This avoids counting the gray bag area as white ink.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Low saturation + medium/high brightness = gray/transparent preview area
    gray_transparent = (S <= 35) & (V >= 120) & (V <= 245)

    return gray_transparent


def make_basic_family_mask(img_rgb, color_name):
    """
    Backup hue-family mask to help catch anti-aliased edges and shaded pixels.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    if color_name == "White":
        return (S <= 18) & (V >= 248)

    if color_name == "Black":
        return V <= 60

    if color_name == "Red":
        return ((H <= 12) | (H >= 168)) & (S >= 40) & (V >= 45)

    if color_name == "Orange":
        return (H > 12) & (H <= 24) & (S >= 40) & (V >= 45)

    if color_name == "Yellow":
        return (H > 24) & (H <= 38) & (S >= 40) & (V >= 45)

    if color_name == "Green":
        return (H > 38) & (H <= 88) & (S >= 30) & (V >= 35)

    if color_name == "Cyan / Blue-Green":
        return (H > 88) & (H <= 100) & (S >= 30) & (V >= 35)

    if color_name == "Blue":
        return (H > 100) & (H <= 140) & (S >= 30) & (V >= 35)

    if color_name == "Purple":
        return (H > 140) & (H <= 158) & (S >= 35) & (V >= 45)

    if color_name == "Pink":
        return (H > 158) & (H < 168) & (S >= 35) & (V >= 45)

    return np.zeros(img_rgb.shape[:2], dtype=bool)


with st.spinner("Calculating ink coverage..."):
    h, w = img_rgb_original.shape[:2]
    scale = min(1.0, (MAX_PIXELS / (h * w)) ** 0.5)

    if scale < 1.0:
        img_rgb = cv2.resize(
            img_rgb_original,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        img_rgb = img_rgb_original.copy()

    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]

    img_rgb_float = img_rgb.astype(np.float32) / 255.0
    img_lab = rgb2lab(img_rgb_float)

    transparent_mask = make_transparent_background_mask(img_rgb)
    printable_area = ~transparent_mask

    priority_order = [
        "Black",
        "White",
        "Red",
        "Orange",
        "Yellow",
        "Green",
        "Cyan / Blue-Green",
        "Blue",
        "Purple",
        "Pink",
    ]

    assigned = np.zeros(img_rgb.shape[:2], dtype=bool)
    final_masks = {}

    for color_name in priority_order:
        if color_name not in selected_colors:
            continue

        target_rgb_float = hex_to_rgb_float(selected_colors[color_name])
        target_lab = rgb2lab(target_rgb_float.reshape(1, 1, 3))[0, 0]

        target_lab_image = np.zeros_like(img_lab)
        target_lab_image[:, :] = target_lab

        delta_e = deltaE_ciede2000(img_lab, target_lab_image)
        delta_mask = delta_e <= DELTA_E_TOLERANCE

        family_mask = make_basic_family_mask(img_rgb, color_name)

        if color_name in ["White", "Black"]:
            color_mask = family_mask
        else:
            color_mask = delta_mask | family_mask

        # Do not count transparent gray area
        color_mask = color_mask & printable_area

        # Avoid double counting
        color_mask = color_mask & (~assigned)

        final_masks[color_name] = color_mask
        assigned |= color_mask

    st.subheader("3️⃣ Ink coverage result")

    results = []
    combined_mask = np.zeros(img_rgb.shape[:2], dtype=bool)

    for color_name, mask in final_masks.items():
        pixels = int(np.sum(mask))
        percent = pixels / total_pixels * 100

        combined_mask |= mask

        results.append(
            {
                "Ink Color": color_name,
                "Coverage %": round(percent, 2),
                "Pixels Counted": pixels,
                "Sample HEX": selected_colors[color_name],
            }
        )

    total_selected = np.sum(combined_mask) / total_pixels * 100

    st.success(f"🎯 Total selected ink coverage: {total_selected:.2f}%")

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    combined_mask_full = cv2.resize(
        combined_mask.astype(np.uint8),
        (img_rgb_original.shape[1], img_rgb_original.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    st.subheader("4️⃣ Selected ink preview")

    selected_preview = img_rgb_original.copy()
    selected_preview[~combined_mask_full] = [225, 225, 225]

    st.image(selected_preview, use_container_width=True)

    st.subheader("5️⃣ Clean detection preview")

    preview_colors = {
        "White": [255, 255, 255],
        "Black": [0, 0, 0],
        "Blue": [30, 45, 150],
        "Green": [0, 140, 70],
        "Red": [200, 30, 40],
        "Yellow": [255, 210, 0],
        "Orange": [240, 120, 30],
        "Purple": [120, 60, 160],
        "Pink": [230, 60, 140],
        "Cyan / Blue-Green": [0, 170, 180],
    }

    clean_preview = np.full_like(img_rgb_original, 230)

    for color_name, mask in final_masks.items():
        mask_full = cv2.resize(
            mask.astype(np.uint8),
            (img_rgb_original.shape[1], img_rgb_original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        clean_preview[mask_full] = preview_colors[color_name]

    st.image(clean_preview, use_container_width=True)

st.info(
    "This version uses CIEDE2000 Delta E color matching, which is better for color accuracy than HSV/RGB matching."
)
