import streamlit as st
import numpy as np
import cv2
import pandas as pd

st.set_page_config(page_title="Ink Coverage Calculator", layout="centered")
st.title("🎨 Ink Coverage Calculator")

MAX_PIXELS = 300000

SPOT_MODE = "Spot Color Mode"
CMYK_MODE = "CMYK Photo Mode (C = Cyan/Blue, M = Magenta/Pink-Red, Y = Yellow, K = Black)"
BOTH_MODE = "Spot Color + CMYK Photo Mode"

st.info(
    "Use Spot Color Mode for normal packaging colors. "
    "Use CMYK Photo Mode when the design contains photos, gradients, or realistic images."
)

# =========================
# Helper functions
# =========================
def resize_for_analysis(img_rgb):
    h, w = img_rgb.shape[:2]
    scale = min(1.0, (MAX_PIXELS / (h * w)) ** 0.5)

    if scale < 1.0:
        return cv2.resize(
            img_rgb,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return img_rgb.copy()


def hex_to_rgb(hex_color):
    hex_color = hex_color.replace("#", "")
    return np.array(
        [
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        ],
        dtype=np.uint8,
    )


def rgb_to_lab(rgb):
    sample = np.uint8([[rgb]])
    lab = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)[0, 0]
    return lab.astype(np.float32)


def image_to_lab(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def make_transparent_mask(img_rgb, mode):
    if mode == "No transparent areas":
        return np.zeros(img_rgb.shape[:2], dtype=bool)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    return (S <= 35) & (V >= 95) & (V <= 225)


def make_visible_white_mask(img_rgb, transparent_mask):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    white = (S <= 22) & (V >= 245)
    return white & (~transparent_mask)


def make_white_panel_backing_mask(img_rgb, transparent_mask):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    near_white = (S <= 40) & (V >= 225)
    panel_mask = near_white & (~transparent_mask)

    panel_u8 = panel_mask.astype(np.uint8) * 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    panel_u8 = cv2.morphologyEx(panel_u8, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    panel_u8 = cv2.morphologyEx(panel_u8, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return panel_u8 > 0


def assign_pixels_to_nearest_ink(img_rgb, selected_colors, printable_area):
    img_lab = image_to_lab(img_rgb)

    ink_names = list(selected_colors.keys())
    distance_stack = []

    for ink in ink_names:
        target_rgb = hex_to_rgb(selected_colors[ink])
        target_lab = rgb_to_lab(target_rgb)

        target_lab_image = np.zeros_like(img_lab)
        target_lab_image[:, :] = target_lab

        distance = np.sqrt(np.sum((img_lab - target_lab_image) ** 2, axis=2))
        distance_stack.append(distance)

    distance_stack = np.stack(distance_stack, axis=0)
    nearest_index = np.argmin(distance_stack, axis=0)

    masks = {}

    for i, ink in enumerate(ink_names):
        masks[ink] = (nearest_index == i) & printable_area

    return masks


def rgb_to_cmyk_coverage(img_rgb, printable_area):
    rgb = img_rgb.astype(np.float32) / 255.0

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    K = 1 - np.maximum.reduce([R, G, B])

    denominator = 1 - K
    denominator = np.where(denominator == 0, 1, denominator)

    C = (1 - R - K) / denominator
    M = (1 - G - K) / denominator
    Y = (1 - B - K) / denominator

    C = np.clip(C, 0, 1)
    M = np.clip(M, 0, 1)
    Y = np.clip(Y, 0, 1)
    K = np.clip(K, 0, 1)

    C[~printable_area] = 0
    M[~printable_area] = 0
    Y[~printable_area] = 0
    K[~printable_area] = 0

    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]

    return {
        "CMYK Cyan (C / Blue)": np.sum(C) / total_pixels * 100,
        "CMYK Magenta (M / Pink-Red)": np.sum(M) / total_pixels * 100,
        "CMYK Yellow (Y)": np.sum(Y) / total_pixels * 100,
        "CMYK Black (K)": np.sum(K) / total_pixels * 100,
    }, C, M, Y, K


# =========================
# Upload image
# =========================
uploaded = st.file_uploader("📤 Upload artwork image", type=["png", "jpg", "jpeg"])

if not uploaded:
    st.info("Upload artwork image first.")
    st.stop()

img_bytes = uploaded.read()
img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("Could not read image.")
    st.stop()

img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb = resize_for_analysis(img_rgb_original)

st.subheader("1️⃣ Original image")
st.image(img_rgb_original, use_container_width=True)

# =========================
# Settings
# =========================
st.subheader("2️⃣ Calculation mode")

calc_mode = st.radio(
    "Choose mode",
    [
        SPOT_MODE,
        CMYK_MODE,
        BOTH_MODE,
    ],
    index=0,
)

transparent_mode = st.radio(
    "Transparent area setting",
    [
        "Gray = transparent",
        "No transparent areas",
    ],
    index=0,
)

transparent_mask = make_transparent_mask(img_rgb, transparent_mode)
printable_area = ~transparent_mask

# =========================
# Spot colors
# =========================
selected_colors = {}

preset_colors = {
    "White": "#ffffff",
    "Black": "#000000",
    "Gray": "#777777",
    "Blue": "#1f2f8f",
    "Green": "#007a5a",
    "Red": "#d71920",
    "Yellow": "#ffd200",
    "Orange": "#f58220",
    "Purple": "#6a1b9a",
    "Pink": "#e91e63",
    "Cyan / Blue-Green": "#00a6b4",
}

if calc_mode in [SPOT_MODE, BOTH_MODE]:
    st.subheader("3️⃣ Tick actual spot ink colors")

    cols = st.columns(3)

    for i, (color_name, default_hex) in enumerate(preset_colors.items()):
        with cols[i % 3]:
            checked = st.checkbox(
                color_name,
                value=False,
                key=f"manual_{color_name}",
            )

            if checked:
                selected_colors[color_name] = st.color_picker(
                    f"{color_name} sample",
                    default_hex,
                    key=f"picker_{color_name}",
                )

# =========================
# White settings
# =========================
st.subheader("4️⃣ White ink settings")

white_underprint = st.checkbox(
    "White underprint exists under colored inks",
    value=True,
)

white_backing_mode = st.radio(
    "White plate method",
    [
        "Visible white only",
        "Visible white + underprint under colored ink",
        "Full white backing behind white panels / label areas",
    ],
    index=2,
)

if not st.button("Calculate Ink Coverage"):
    st.info("Click Calculate after choosing settings.")
    st.stop()

# =========================
# Calculate
# =========================
with st.spinner("Calculating ink coverage..."):
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]

    results = []

    combined_spot_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
    visible_colored_mask = np.zeros(img_rgb.shape[:2], dtype=bool)

    if calc_mode in [SPOT_MODE, BOTH_MODE]:
        if not selected_colors:
            st.error("Please tick at least one spot ink color.")
            st.stop()

        assigned_masks = assign_pixels_to_nearest_ink(
            img_rgb,
            selected_colors,
            printable_area,
        )

        for ink, mask in assigned_masks.items():
            percent = np.sum(mask) / total_pixels * 100
            combined_spot_mask |= mask

            if ink != "White":
                visible_colored_mask |= mask

            results.append(
                {
                    "Ink / Plate": ink,
                    "Coverage %": round(percent, 2),
                    "Type": "Spot Color",
                }
            )

        visible_white_mask = make_visible_white_mask(img_rgb, transparent_mask)

        if white_backing_mode == "Visible white only":
            white_plate_mask = visible_white_mask

        elif white_backing_mode == "Visible white + underprint under colored ink":
            if white_underprint:
                white_plate_mask = visible_white_mask | visible_colored_mask
            else:
                white_plate_mask = visible_white_mask

        else:
            white_panel_mask = make_white_panel_backing_mask(img_rgb, transparent_mask)
            if white_underprint:
                white_plate_mask = white_panel_mask | visible_colored_mask
            else:
                white_plate_mask = white_panel_mask

        white_plate_percent = np.sum(white_plate_mask) / total_pixels * 100

        results.append(
            {
                "Ink / Plate": "White plate final",
                "Coverage %": round(white_plate_percent, 2),
                "Type": "White Underprint / Backing",
            }
        )

    else:
        white_plate_mask = make_visible_white_mask(img_rgb, transparent_mask)

    if calc_mode in [CMYK_MODE, BOTH_MODE]:
        cmyk_results, C, M, Y, K = rgb_to_cmyk_coverage(img_rgb, printable_area)

        for plate, percent in cmyk_results.items():
            results.append(
                {
                    "Ink / Plate": plate,
                    "Coverage %": round(percent, 2),
                    "Type": "CMYK Estimate",
                }
            )

    df = pd.DataFrame(results)

    st.subheader("5️⃣ Ink coverage result")
    st.dataframe(df, use_container_width=True)

    total_laydown = df["Coverage %"].sum()
    st.success(f"Total ink laydown estimate: {total_laydown:.2f}%")

    st.caption(
        "For photo/gradient designs, CMYK is an RGB-based estimate. "
        "For exact CMYK ink %, use the original AI/PDF separation file."
    )

    transparent_full = cv2.resize(
        transparent_mask.astype(np.uint8),
        (img_rgb_original.shape[1], img_rgb_original.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    if calc_mode in [SPOT_MODE, BOTH_MODE]:
        combined_spot_full = cv2.resize(
            combined_spot_mask.astype(np.uint8),
            (img_rgb_original.shape[1], img_rgb_original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        white_plate_full = cv2.resize(
            white_plate_mask.astype(np.uint8),
            (img_rgb_original.shape[1], img_rgb_original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        st.subheader("6️⃣ Spot ink preview")

        spot_preview = img_rgb_original.copy()
        spot_preview[~combined_spot_full] = [225, 225, 225]
        spot_preview[transparent_full] = [180, 180, 180]

        st.image(spot_preview, use_container_width=True)

        st.subheader("7️⃣ White plate preview")

        white_preview = np.full_like(img_rgb_original, 225)
        white_preview[white_plate_full] = [255, 255, 255]
        white_preview[transparent_full] = [180, 180, 180]

        st.image(white_preview, use_container_width=True)

    if calc_mode in [CMYK_MODE, BOTH_MODE]:
        st.subheader("8️⃣ CMYK photo estimate preview")

        cmyk_preview = img_rgb_original.copy()
        cmyk_preview[transparent_full] = [180, 180, 180]
        st.image(cmyk_preview, use_container_width=True)

st.info(
    "Use Spot Color Mode for flat color jobs. "
    "Use CMYK Photo Mode when photos or gradients are printed. "
    "Use Spot Color + CMYK when the design has both spot colors and photo areas."
)
