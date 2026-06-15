import streamlit as st
import numpy as np
import cv2
import pandas as pd
from skimage.color import rgb2lab, deltaE_ciede2000

st.set_page_config(page_title="Ink Coverage Calculator", layout="centered")
st.title("🎨 Ink Coverage Calculator")

MAX_PIXELS = 300000
AUTO_COLOR_COUNT = 6

st.info(
    "Ink % is based on selected print colors. Gray can be treated as transparent/no ink. "
    "White underprint and white backing are calculated separately."
)

# =========================
# Helper functions
# =========================
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


def rgb_to_hex(rgb):
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def approximate_color_name(rgb):
    sample = np.uint8([[rgb]])
    hsv = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = hsv

    if s <= 25 and v >= 245:
        return "White"
    if v <= 60:
        return "Black"
    if s <= 45:
        return "Gray"
    if h <= 12 or h >= 168:
        return "Red"
    if h <= 24:
        return "Orange"
    if h <= 38:
        return "Yellow"
    if h <= 88:
        return "Green"
    if h <= 100:
        return "Cyan / Blue-Green"
    if h <= 140:
        return "Blue"
    if h <= 158:
        return "Purple"
    return "Pink"


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


def make_transparent_mask(img_rgb, mode):
    if mode == "No transparent areas":
        return np.zeros(img_rgb.shape[:2], dtype=bool)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    if mode == "Gray = transparent":
        return (S <= 35) & (V >= 95) & (V <= 225)

    # Auto detect transparent gray
    return (S <= 35) & (V >= 100) & (V <= 230)


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


def auto_detect_colors(img_rgb, transparent_mask, k=AUTO_COLOR_COUNT):
    valid_pixels = img_rgb[~transparent_mask]

    if len(valid_pixels) < 100:
        valid_pixels = img_rgb.reshape(-1, 3)

    pixels = valid_pixels.astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.2,
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = centers.astype(np.uint8)
    counts = np.bincount(labels.flatten(), minlength=k)
    percents = counts / counts.sum() * 100

    detected = []

    for i in range(k):
        rgb = centers[i]
        detected.append(
            {
                "name": approximate_color_name(rgb),
                "hex": rgb_to_hex(rgb),
                "percent": float(percents[i]),
                "rgb": rgb,
            }
        )

    detected.sort(key=lambda x: x["percent"], reverse=True)
    return detected


def assign_pixels_to_nearest_ink(img_rgb, selected_colors, printable_area):
    img_float = img_rgb.astype(np.float32) / 255.0
    img_lab = rgb2lab(img_float)

    ink_names = list(selected_colors.keys())
    distance_stack = []

    for ink in ink_names:
        target_rgb = hex_to_rgb_float(selected_colors[ink])
        target_lab = rgb2lab(target_rgb.reshape(1, 1, 3))[0, 0]

        target_lab_image = np.zeros_like(img_lab)
        target_lab_image[:, :] = target_lab

        delta_e = deltaE_ciede2000(img_lab, target_lab_image)
        distance_stack.append(delta_e)

    distance_stack = np.stack(distance_stack, axis=0)

    nearest_index = np.argmin(distance_stack, axis=0)

    masks = {}

    for i, ink in enumerate(ink_names):
        masks[ink] = (nearest_index == i) & printable_area

    return masks


# =========================
# Upload image first
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
st.subheader("2️⃣ Transparent area setting")

transparent_mode = st.radio(
    "How should gray areas be treated?",
    [
        "Gray = transparent",
        "No transparent areas",
        "Auto detect transparent gray",
    ],
    index=0,
)

transparent_mask = make_transparent_mask(img_rgb, transparent_mode)
printable_area = ~transparent_mask

# =========================
# Color selection mode
# =========================
st.subheader("3️⃣ Color selection method")

selection_mode = st.radio(
    "Choose method",
    [
        "Manual: tick actual ink colors",
        "Auto-detect: program finds main colors",
    ],
    index=0,
)

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

if selection_mode == "Manual: tick actual ink colors":
    st.write("Tick only actual printing inks.")

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
                    key=f"picker_manual_{color_name}",
                )

else:
    detected_colors = auto_detect_colors(img_rgb, transparent_mask, AUTO_COLOR_COUNT)

    st.write(f"Program detected **{len(detected_colors)} main colors**. Untick colors that are not ink.")

    for i, color in enumerate(detected_colors):
        label = f"{color['name']} {color['hex']} — approx {color['percent']:.1f}%"

        col1, col2 = st.columns([1, 5])

        with col1:
            st.markdown(
                f"""
                <div style="
                    width:34px;
                    height:34px;
                    background:{color['hex']};
                    border:1px solid #000;
                    border-radius:4px;">
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            checked = st.checkbox(
                label,
                value=color["name"] != "Gray",
                key=f"auto_check_{i}",
            )

            if checked:
                ink_name = f"{color['name']} #{i + 1}"
                selected_colors[ink_name] = st.color_picker(
                    f"Adjust {ink_name}",
                    color["hex"],
                    key=f"auto_picker_{i}",
                )

if not selected_colors:
    st.warning("Please select at least one ink color.")
    st.stop()

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

st.caption(
    "For transparent CPP/PE/OPP, white usually prints under colored areas. "
    "Use full white backing when white label panels are printed as solid white blocks."
)

if not st.button("Calculate Ink Coverage"):
    st.info("Click Calculate after checking the colors and white settings.")
    st.stop()

# =========================
# Calculate
# =========================
with st.spinner("Calculating ink coverage..."):
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]

    assigned_masks = assign_pixels_to_nearest_ink(
        img_rgb,
        selected_colors,
        printable_area,
    )

    visible_white_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
    visible_colored_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
    combined_visible_mask = np.zeros(img_rgb.shape[:2], dtype=bool)

    for ink, mask in assigned_masks.items():
        combined_visible_mask |= mask

        if "White" in ink:
            visible_white_mask |= mask
        else:
            visible_colored_mask |= mask

    strict_visible_white_mask = make_visible_white_mask(img_rgb, transparent_mask)

    if white_backing_mode == "Visible white only":
        white_plate_mask = strict_visible_white_mask

    elif white_backing_mode == "Visible white + underprint under colored ink":
        if white_underprint:
            white_plate_mask = strict_visible_white_mask | visible_colored_mask
        else:
            white_plate_mask = strict_visible_white_mask

    else:
        white_panel_backing_mask = make_white_panel_backing_mask(img_rgb, transparent_mask)

        if white_underprint:
            white_plate_mask = white_panel_backing_mask | visible_colored_mask
        else:
            white_plate_mask = white_panel_backing_mask

    visible_colored_percent = np.sum(visible_colored_mask) / total_pixels * 100
    white_plate_percent = np.sum(white_plate_mask) / total_pixels * 100
    total_visible_percent = np.sum(combined_visible_mask) / total_pixels * 100
    total_ink_laydown = visible_colored_percent + white_plate_percent

    st.subheader("5️⃣ Ink coverage result")

    results = []

    for ink, mask in assigned_masks.items():
        percent = np.sum(mask) / total_pixels * 100

        results.append(
            {
                "Ink / Plate": ink,
                "Coverage %": round(percent, 2),
                "Pixels Counted": int(np.sum(mask)),
                "Sample HEX": selected_colors[ink],
            }
        )

    results.append(
        {
            "Ink / Plate": "White plate final",
            "Coverage %": round(white_plate_percent, 2),
            "Pixels Counted": int(np.sum(white_plate_mask)),
            "Sample HEX": "#ffffff",
        }
    )

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    st.success(f"Visible printed area: {total_visible_percent:.2f}%")
    st.success(f"White plate coverage: {white_plate_percent:.2f}%")
    st.success(f"Total ink laydown estimate: {total_ink_laydown:.2f}%")

    st.caption(
        "Use individual plate % for each ink. Use total ink laydown only for overall ink-load comparison."
    )

    # =========================
    # Full-resolution previews
    # =========================
    combined_visible_full = cv2.resize(
        combined_visible_mask.astype(np.uint8),
        (img_rgb_original.shape[1], img_rgb_original.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    white_plate_full = cv2.resize(
        white_plate_mask.astype(np.uint8),
        (img_rgb_original.shape[1], img_rgb_original.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    transparent_full = cv2.resize(
        transparent_mask.astype(np.uint8),
        (img_rgb_original.shape[1], img_rgb_original.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    st.subheader("6️⃣ Selected printable ink preview")

    selected_preview = img_rgb_original.copy()
    selected_preview[~combined_visible_full] = [225, 225, 225]
    selected_preview[transparent_full] = [180, 180, 180]

    st.image(selected_preview, use_container_width=True)

    st.subheader("7️⃣ White plate preview")

    white_preview = np.full_like(img_rgb_original, 225)
    white_preview[white_plate_full] = [255, 255, 255]
    white_preview[transparent_full] = [180, 180, 180]

    st.image(white_preview, use_container_width=True)

    st.subheader("8️⃣ Clean ink assignment preview")

    clean_preview = np.full_like(img_rgb_original, 230)

    preview_palette = [
        [255, 255, 255],
        [0, 0, 0],
        [140, 140, 140],
        [30, 45, 150],
        [0, 140, 70],
        [200, 30, 40],
        [255, 210, 0],
        [240, 120, 30],
        [120, 60, 160],
        [230, 60, 140],
        [0, 170, 180],
    ]

    for i, (ink, mask) in enumerate(assigned_masks.items()):
        mask_full = cv2.resize(
            mask.astype(np.uint8),
            (img_rgb_original.shape[1], img_rgb_original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        clean_preview[mask_full] = preview_palette[i % len(preview_palette)]

    clean_preview[transparent_full] = [180, 180, 180]

    st.image(clean_preview, use_container_width=True)

st.info(
    "For highest accuracy: Manual mode is best when you know the real print inks. "
    "Auto-detect is useful when staff does not know how many main colors are in the artwork."
)
