import streamlit as st
import numpy as np
import cv2
import pandas as pd

st.set_page_config(page_title="Ink Coverage Calculator", layout="centered")
st.title("🎨 Ink Coverage Calculator")

# =========================
# 1. Select ink colors first
# =========================
st.subheader("1️⃣ Tick ink colors used")

color_options = [
    "Blue",
    "White",
    "Black",
    "Gray",
    "Red",
    "Green",
    "Yellow",
    "Orange",
    "Purple",
    "Pink",
    "Cyan / Blue-Green",
]

selected_colors = []

cols = st.columns(3)

for i, color in enumerate(color_options):
    with cols[i % 3]:
        default_on = color in ["Blue", "White"]
        if st.checkbox(color, value=default_on, key=f"check_{color}"):
            selected_colors.append(color)

if not selected_colors:
    st.warning("Please tick at least one ink color.")
    st.stop()

# =========================
# 2. Upload image
# =========================
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

# =========================
# 3. Fixed best settings
# =========================
MAX_PIXELS = 800000

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

# =========================
# 4. Convert to HSV
# =========================
hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

H = hsv[:, :, 0]
S = hsv[:, :, 1]
V = hsv[:, :, 2]

total_pixels = H.size

# =========================
# 5. Detect color families
# =========================
masks = {}

# Neutral colors
masks["White"] = (S <= 45) & (V >= 165)
masks["Black"] = V <= 65
masks["Gray"] = (S <= 55) & (V > 65) & (V < 165)

# Ink color families
masks["Red"] = ((H <= 12) | (H >= 168)) & (S >= 35) & (V >= 45)
masks["Orange"] = (H > 12) & (H <= 24) & (S >= 35) & (V >= 45)
masks["Yellow"] = (H > 24) & (H <= 38) & (S >= 35) & (V >= 45)
masks["Green"] = (H > 38) & (H <= 88) & (S >= 35) & (V >= 45)
masks["Cyan / Blue-Green"] = (H > 88) & (H <= 100) & (S >= 35) & (V >= 45)
masks["Blue"] = (H > 100) & (H <= 140) & (S >= 30) & (V >= 35)
masks["Purple"] = (H > 140) & (H <= 158) & (S >= 35) & (V >= 45)
masks["Pink"] = (H > 158) & (H < 168) & (S >= 35) & (V >= 45)

# =========================
# 6. Avoid double counting
# Priority matters: black/white/gray first, then colors
# =========================
priority_order = [
    "Black",
    "White",
    "Gray",
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Cyan / Blue-Green",
    "Blue",
    "Purple",
    "Pink",
]

assigned = np.zeros(H.shape, dtype=bool)
clean_masks = {}

for color in priority_order:
    clean_masks[color] = masks[color] & (~assigned)
    assigned |= clean_masks[color]

# =========================
# 7. Calculate selected coverage
# =========================
st.subheader("3️⃣ Ink coverage result")

results = []
combined_mask = np.zeros(H.shape, dtype=bool)

for color in selected_colors:
    mask = clean_masks[color]
    pixels = np.sum(mask)
    percent = pixels / total_pixels * 100

    combined_mask |= mask

    results.append(
        {
            "Ink Color": color,
            "Coverage %": round(percent, 2),
            "Pixels Counted": int(pixels),
        }
    )

total_selected = np.sum(combined_mask) / total_pixels * 100

st.success(f"🎯 Total selected ink coverage: {total_selected:.2f}%")

df = pd.DataFrame(results)
st.dataframe(df, use_container_width=True)

# =========================
# 8. Full color family breakdown
# =========================
st.subheader("4️⃣ Full detected color breakdown")

all_results = []

for color in priority_order:
    pixels = np.sum(clean_masks[color])
    percent = pixels / total_pixels * 100

    all_results.append(
        {
            "Color Family": color,
            "Coverage %": round(percent, 2),
        }
    )

df_all = pd.DataFrame(all_results)
st.dataframe(df_all, use_container_width=True)

# =========================
# 9. Preview selected colors only
# =========================
st.subheader("5️⃣ Selected ink preview")

preview = img_rgb.copy()
preview[~combined_mask] = [220, 220, 220]

st.image(preview, use_container_width=True)

# =========================
# 10. Simplified detected preview
# =========================
st.subheader("6️⃣ Color-family detection preview")

family_preview = np.full_like(img_rgb, 230)

preview_colors = {
    "White": [255, 255, 255],
    "Black": [0, 0, 0],
    "Gray": [150, 150, 150],
    "Blue": [30, 45, 150],
    "Green": [0, 140, 70],
    "Red": [200, 30, 40],
    "Yellow": [255, 210, 0],
    "Orange": [240, 120, 30],
    "Purple": [120, 60, 160],
    "Pink": [230, 60, 140],
    "Cyan / Blue-Green": [0, 170, 180],
}

for color in priority_order:
    family_preview[clean_masks[color]] = preview_colors[color]

st.image(family_preview, use_container_width=True)

st.info(
    "This version groups all similar shades into main ink color families, "
    "so blue text, blue bars, and anti-aliased blue edges are counted together."
)