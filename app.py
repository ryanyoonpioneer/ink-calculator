
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    st.set_page_config(page_title="Ink Coverage Calculator", layout="wide")
    st.error(
        "Missing required package.\n\n"
        "Install once in Command Prompt:\n"
        "pip install streamlit-image-coordinates\n\n"
        "Then run again."
    )
    st.stop()

# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="Ink Coverage Calculator", layout="wide")

MAX_PIXELS = 550_000
DISPLAY_WIDTH = 1100

st.title("🎨 Ink Coverage Calculator for Packaging")
st.caption("Goal: estimate ink % for quotation from artwork images/screenshots. Spot colors are counted outside photo boxes. CMYK is counted only inside photo boxes.")

# =========================================================
# SESSION
# =========================================================
defaults = {
    "photo_boxes": [],
    "first_corner": None,
    "last_click": None,
    "show_advanced": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def clear_boxes():
    st.session_state.photo_boxes = []
    st.session_state.first_corner = None
    st.session_state.last_click = None

# =========================================================
# IMAGE HELPERS
# =========================================================
def resize_for_analysis(img_rgb):
    h, w = img_rgb.shape[:2]
    scale = min(1.0, (MAX_PIXELS / (h * w)) ** 0.5)
    if scale < 1.0:
        return cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_rgb.copy()

def hex_to_rgb(hex_color):
    hex_color = hex_color.replace("#", "")
    return np.array([int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)], dtype=np.uint8)

def rgb_to_lab_color(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

def rgb_to_lab_image(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

def color_distance_lab(img_rgb, target_rgb):
    img_lab = rgb_to_lab_image(img_rgb)
    target_lab = rgb_to_lab_color(target_rgb)
    return np.sqrt(np.sum((img_lab - target_lab) ** 2, axis=2))

def make_background_mask(img_rgb, enabled, bg_hex, tolerance):
    if not enabled:
        return np.zeros(img_rgb.shape[:2], dtype=bool)
    bg_rgb = hex_to_rgb(bg_hex)
    dist = color_distance_lab(img_rgb, bg_rgb)
    return dist <= tolerance

def make_edge_connected_background_mask(img_rgb, enabled, bg_hex, tolerance):
    """
    Smarter background removal for artwork screenshots.
    It removes only background-colored areas connected to the outside edge of the image.

    This prevents white label panels / text boxes inside the artwork from being removed
    just because they are slightly gray or close to the selected background color.
    """
    if not enabled:
        return np.zeros(img_rgb.shape[:2], dtype=bool)

    candidate = make_background_mask(img_rgb, True, bg_hex, tolerance).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(candidate, connectivity=8)

    if num_labels <= 1:
        return candidate.astype(bool)

    edge_labels = set()

    # collect components touching the outside image edge
    edge_labels.update(np.unique(labels[0, :]).tolist())
    edge_labels.update(np.unique(labels[-1, :]).tolist())
    edge_labels.update(np.unique(labels[:, 0]).tolist())
    edge_labels.update(np.unique(labels[:, -1]).tolist())

    edge_labels.discard(0)

    mask = np.isin(labels, list(edge_labels))
    return mask

def make_visible_white_mask(img_rgb, transparent_mask):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    return ((S <= 28) & (V >= 238)) & (~transparent_mask)

def make_white_panel_backing_mask(img_rgb, transparent_mask):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((S <= 45) & (V >= 220)) & (~transparent_mask)
    u8 = mask.astype(np.uint8) * 255
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)), iterations=2)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    cleaned = np.zeros_like(u8)
    min_area = max(80, int(img_rgb.shape[0] * img_rgb.shape[1] * 0.001))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned > 0

def make_rectangle_mask(img_shape, rectangles_percent):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for x1p, y1p, x2p, y2p in rectangles_percent:
        x1, x2 = sorted([int(w * x1p / 100), int(w * x2p / 100)])
        y1, y2 = sorted([int(h * y1p / 100), int(h * y2p / 100)])
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask

def draw_boxes(img_pil, boxes, current_point=None):
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i, (x1p, y1p, x2p, y2p) in enumerate(boxes):
        x1, x2 = sorted([int(w * x1p / 100), int(w * x2p / 100)])
        y1, y2 = sorted([int(h * y1p / 100), int(h * y2p / 100)])
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=5)
        draw.rectangle([x1, max(0, y1-28), x1+105, y1], fill="white", outline="blue")
        draw.text((x1+6, max(0, y1-24)), f"PHOTO {i+1}", fill="blue")
    if current_point is not None:
        x, y = current_point
        r = 8
        draw.ellipse([x-r, y-r, x+r, y+r], fill="red", outline="white", width=2)
        draw.text((x+12, y+8), "1st corner", fill="red")
    return img

def overlay_mask_preview(img_rgb, mask, transparent_mask=None):
    preview = img_rgb.copy()
    preview[~mask] = [230, 230, 230]
    if transparent_mask is not None:
        preview[transparent_mask] = [175, 175, 175]
    return preview

def red_transparency_preview(img_rgb, transparent_mask):
    """
    Live preview for tolerance:
    Red overlay = pixels that will be removed / treated as transparent.
    Normal image = pixels that will still be counted as printable.
    """
    preview = img_rgb.copy()
    overlay = preview.copy()
    overlay[transparent_mask] = [255, 0, 0]
    blended = cv2.addWeighted(preview, 0.68, overlay, 0.32, 0)
    return blended

# =========================================================
# CALC HELPERS
# =========================================================
def assign_pixels_to_spot_inks(img_rgb, selected_colors, printable_area, exclude_mask):
    img_lab = rgb_to_lab_image(img_rgb)
    names = list(selected_colors.keys())
    stack = []
    for name in names:
        target_lab = rgb_to_lab_color(hex_to_rgb(selected_colors[name]))
        stack.append(np.sqrt(np.sum((img_lab - target_lab) ** 2, axis=2)))
    nearest = np.argmin(np.stack(stack, axis=0), axis=0)
    valid = printable_area & (~exclude_mask)
    return {name: (nearest == i) & valid for i, name in enumerate(names)}

def spot_coverage_percent(img_rgb, mask, ink_hex, tint_weighting):
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    if not tint_weighting:
        return np.sum(mask) / total_pixels * 100
    target = hex_to_rgb(ink_hex).astype(np.float32)
    white = np.array([255, 255, 255], dtype=np.float32)
    pixels = img_rgb.astype(np.float32)
    numerator = np.sum((white - pixels) * (white - target), axis=2)
    denominator = np.sum((white - target) * (white - target))
    strength = np.clip(numerator / max(denominator, 1), 0, 1)
    return np.sum(strength[mask]) / total_pixels * 100

def rgb_to_cmyk_coverage(img_rgb, area, method="GCR / UCR print-style CMYK"):
    """
    Estimate CMYK from RGB.

    Simple RGB-to-CMYK:
    - Direct mathematical conversion.
    - Can over-allocate neutral gray into C/M/Y.

    GCR / UCR print-style CMYK:
    - Moves neutral gray/dark component into K black.
    - Usually gives a more realistic printing estimate for photo boxes.
    - Still an estimate. Exact CMYK requires original AI/PDF separations.
    """
    rgb = img_rgb.astype(np.float32) / 255.0
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    if method == "Simple RGB-to-CMYK":
        K = 1 - np.maximum.reduce([R, G, B])
        denom = np.where((1 - K) == 0, 1, 1 - K)
        C = np.clip((1 - R - K) / denom, 0, 1)
        M = np.clip((1 - G - K) / denom, 0, 1)
        Y = np.clip((1 - B - K) / denom, 0, 1)
        K = np.clip(K, 0, 1)
    else:
        # First calculate raw ink demand from RGB.
        C0 = 1 - R
        M0 = 1 - G
        Y0 = 1 - B

        # Neutral component common to C/M/Y.
        neutral = np.minimum.reduce([C0, M0, Y0])

        # More GCR for dark / neutral pixels, less for colorful pixels.
        max_rgb = np.maximum.reduce([R, G, B])
        min_rgb = np.minimum.reduce([R, G, B])
        chroma = max_rgb - min_rgb
        darkness = 1 - max_rgb

        # gcr_strength:
        # - high when dark/neutral
        # - lower when colorful
        gcr_strength = 0.55 + 0.35 * darkness - 0.25 * chroma
        gcr_strength = np.clip(gcr_strength, 0.35, 0.90)

        K = neutral * gcr_strength

        # Remove the neutral amount shifted into K from CMY.
        C = C0 - K
        M = M0 - K
        Y = Y0 - K

        # UCR: reduce extra CMY in very dark areas where K should dominate.
        ucr = np.clip((darkness - 0.45) / 0.55, 0, 1) * 0.25
        C *= (1 - ucr)
        M *= (1 - ucr)
        Y *= (1 - ucr)

        C = np.clip(C, 0, 1)
        M = np.clip(M, 0, 1)
        Y = np.clip(Y, 0, 1)
        K = np.clip(K, 0, 1)

    C[~area], M[~area], Y[~area], K[~area] = 0, 0, 0, 0

    total = img_rgb.shape[0] * img_rgb.shape[1]
    photo_area_pct = np.sum(area) / total * 100
    total_cmyk_laydown = (np.sum(C) + np.sum(M) + np.sum(Y) + np.sum(K)) / total * 100

    return {
        "CMYK Cyan": np.sum(C) / total * 100,
        "CMYK Magenta": np.sum(M) / total * 100,
        "CMYK Yellow": np.sum(Y) / total * 100,
        "CMYK Black": np.sum(K) / total * 100,
        "CMYK photo area": photo_area_pct,
        "CMYK total laydown": total_cmyk_laydown,
    }

# =========================================================
# UPLOAD
# =========================================================
uploaded = st.file_uploader("1️⃣ Upload artwork image", type=["png", "jpg", "jpeg"])

if not uploaded:
    st.info("Upload an artwork image to start.")
    st.stop()

img_bytes = uploaded.read()
img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Could not read image.")
    st.stop()

img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb = resize_for_analysis(img_rgb_original)

orig_h, orig_w = img_rgb_original.shape[:2]
display_w = min(DISPLAY_WIDTH, orig_w)
display_h = int(orig_h * display_w / orig_w)
display_img = Image.fromarray(img_rgb_original).resize((display_w, display_h))

# =========================================================
# STEP 2 JOB TYPE
# =========================================================
st.subheader("2️⃣ Choose job type")

job_type = st.radio(
    "What kind of artwork is this?",
    [
        "Spot colors only — no photos",
        "Spot colors + photo boxes",
        "CMYK photo only",
    ],
    index=1,
    horizontal=True,
)

has_spot = job_type in ["Spot colors only — no photos", "Spot colors + photo boxes"]
has_cmyk = job_type in ["Spot colors + photo boxes", "CMYK photo only"]

# =========================================================
# STEP 3 BACKGROUND / TRANSPARENT
# =========================================================
with st.expander("3️⃣ Background / transparent area", expanded=True):
    bg_left, bg_right = st.columns([1, 1.4])

    with bg_left:
        bg_method = st.radio(
            "Transparent/background method",
            [
                "Smart edge background removal - recommended",
                "Simple color tolerance",
                "No transparent background removal",
            ],
            index=0,
        )

        remove_bg = bg_method != "No transparent background removal"

        bg_hex = st.color_picker("Background color to remove", "#c7c7c7")
        bg_tol = st.slider(
            "Tolerance",
            1,
            60,
            18,
            help="Higher removes more pixels close to the selected background color. Watch the red preview.",
        )

        st.caption(
            "Recommended mode only removes background-colored areas connected to the outside edge. "
            "This helps protect white label/text panels inside the artwork."
        )

    if bg_method == "Smart edge background removal - recommended":
        transparent_mask = make_edge_connected_background_mask(img_rgb, remove_bg, bg_hex, bg_tol)
    elif bg_method == "Simple color tolerance":
        transparent_mask = make_background_mask(img_rgb, remove_bg, bg_hex, bg_tol)
    else:
        transparent_mask = np.zeros(img_rgb.shape[:2], dtype=bool)

    printable_area = ~transparent_mask

    removed_pct = np.sum(transparent_mask) / transparent_mask.size * 100
    printable_pct = 100 - removed_pct

    with bg_right:
        st.markdown(
            f"**Live tolerance preview:** "
            f"<span style='color:red;'>{removed_pct:.1f}% removed</span> · "
            f"{printable_pct:.1f}% printable",
            unsafe_allow_html=True,
        )
        st.image(
            red_transparency_preview(img_rgb, transparent_mask),
            caption="Red overlay = transparent / no ink. White panels/text should stay normal, not red.",
            use_container_width=True,
        )

# =========================================================
# STEP 4 CMYK PHOTO BOX SELECTION
# =========================================================
cmyk_area = np.zeros(img_rgb.shape[:2], dtype=bool)

if has_cmyk:
    st.subheader("4️⃣ Select photo boxes for CMYK")
    st.caption("Click one corner of a photo, then click the opposite corner. Do this only for real photo/gradient areas.")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 4])
    with c1:
        if st.button("Clear boxes"):
            clear_boxes()
            st.rerun()
    with c2:
        if st.button("Undo last"):
            if st.session_state.photo_boxes:
                st.session_state.photo_boxes.pop()
            st.rerun()
    with c3:
        if st.button("Cancel corner"):
            st.session_state.first_corner = None
            st.session_state.last_click = None
            st.rerun()

    if st.session_state.first_corner is None:
        st.info("Next click: first corner of a photo box.")
    else:
        st.warning("Next click: opposite corner.")

    selector_img = draw_boxes(display_img, st.session_state.photo_boxes, st.session_state.first_corner)
    click = streamlit_image_coordinates(selector_img, key="photo_box_selector")

    if click is not None:
        x, y = int(click["x"]), int(click["y"])
        click_id = (x, y)
        if click_id != st.session_state.last_click:
            st.session_state.last_click = click_id
            if st.session_state.first_corner is None:
                st.session_state.first_corner = (x, y)
                st.rerun()
            else:
                x1, y1 = st.session_state.first_corner
                if abs(x-x1) > 5 and abs(y-y1) > 5:
                    st.session_state.photo_boxes.append((
                        x1 / display_w * 100,
                        y1 / display_h * 100,
                        x / display_w * 100,
                        y / display_h * 100,
                    ))
                st.session_state.first_corner = None
                st.rerun()

    cmyk_area = make_rectangle_mask(img_rgb.shape, st.session_state.photo_boxes) & printable_area
    st.success(f"Selected photo boxes: {len(st.session_state.photo_boxes)}")
else:
    st.subheader("4️⃣ Artwork preview")
    st.image(img_rgb_original, use_container_width=True)

# =========================================================
# STEP 5 PREVIEWS
# =========================================================
with st.expander("Preview selected CMYK/photo area", expanded=True):
    if has_cmyk:
        st.image(
            overlay_mask_preview(img_rgb, cmyk_area, transparent_mask),
            caption="CMYK photo area preview: normal colors = selected CMYK area",
            use_container_width=True,
        )
    else:
        st.info("No CMYK/photo boxes for this job type.")

# =========================================================
# STEP 6 INKS
# =========================================================
st.subheader("5️⃣ Select actual inks")

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

ink_col, white_col = st.columns([1.3, 1])

with ink_col:
    if has_spot:
        st.markdown("**Spot inks**")
        st.caption("Tick only inks that will be printed as spot plates. For photos, do not tick CMYK colors here.")
        cols = st.columns(4)
        for i, (name, default_hex) in enumerate(preset_colors.items()):
            with cols[i % 4]:
                checked = st.checkbox(name, value=(name in ["White", "Black"]), key=f"ink_{name}")
                if checked:
                    selected_colors[name] = st.color_picker(f"{name} sample", default_hex, key=f"color_{name}")

        st.markdown("**Custom spot inks**")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            n1 = st.text_input("Custom ink 1 name", "")
            c1v = st.color_picker("Custom ink 1 color", "#007a5a")
        with cc2:
            n2 = st.text_input("Custom ink 2 name", "")
            c2v = st.color_picker("Custom ink 2 color", "#d71920")
        with cc3:
            n3 = st.text_input("Custom ink 3 name", "")
            c3v = st.color_picker("Custom ink 3 color", "#777777")
        if n1.strip(): selected_colors[n1.strip()] = c1v
        if n2.strip(): selected_colors[n2.strip()] = c2v
        if n3.strip(): selected_colors[n3.strip()] = c3v
    else:
        st.info("Spot ink selection is off for CMYK photo only mode.")

with white_col:
    st.markdown("**White plate logic**")
    white_underprint = st.checkbox("White underprint exists under colored spot inks", value=True)
    white_method = st.radio(
        "White method",
        [
            "Visible white only",
            "Visible white + underprint under colored ink",
            "Full white backing behind white panels / label areas",
        ],
        index=2,
    )
    white_under_cmyk = False
    cmyk_method = "GCR / UCR print-style CMYK"
    if has_cmyk:
        white_under_cmyk = st.checkbox("White underprint also under CMYK photo boxes", value=True)
        cmyk_method = st.radio(
            "CMYK breakdown method",
            [
                "GCR / UCR print-style CMYK",
                "Simple RGB-to-CMYK",
            ],
            index=0,
            help="GCR/UCR moves neutral gray/dark photo areas into K black, which is usually more realistic for printing."
        )
    tint_weighting = st.checkbox("Count light/medium spot colors as tint screens", value=True)

# =========================================================
# CALCULATE
# =========================================================
st.subheader("6️⃣ Calculate")
calculate = st.button("Calculate Ink Coverage", type="primary", use_container_width=True)

if not calculate:
    st.stop()

results = []
total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
combined_spot_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
visible_colored_mask = np.zeros(img_rgb.shape[:2], dtype=bool)

if has_spot:
    if not selected_colors:
        st.error("Please select at least one spot ink.")
        st.stop()

    exclude_from_spot = cmyk_area if has_cmyk else np.zeros(img_rgb.shape[:2], dtype=bool)
    assigned = assign_pixels_to_spot_inks(img_rgb, selected_colors, printable_area, exclude_from_spot)

    for ink, mask in assigned.items():
        if ink == "White":
            pct = np.sum(mask) / total_pixels * 100
        else:
            pct = spot_coverage_percent(img_rgb, mask, selected_colors[ink], tint_weighting)
        combined_spot_mask |= mask
        if ink != "White":
            visible_colored_mask |= mask
        results.append({"Ink / Plate": ink, "Coverage %": round(pct, 2), "Type": "Spot Color"})

visible_white_mask = make_visible_white_mask(img_rgb, transparent_mask)

if white_method == "Visible white only":
    white_plate_mask = visible_white_mask
elif white_method == "Visible white + underprint under colored ink":
    white_plate_mask = visible_white_mask | visible_colored_mask if white_underprint else visible_white_mask
else:
    white_panel_mask = make_white_panel_backing_mask(img_rgb, transparent_mask)
    white_plate_mask = white_panel_mask | visible_colored_mask if white_underprint else white_panel_mask

if white_under_cmyk:
    white_plate_mask = white_plate_mask | cmyk_area

white_pct = np.sum(white_plate_mask) / total_pixels * 100
results.append({"Ink / Plate": "White plate final", "Coverage %": round(white_pct, 2), "Type": "White Underprint / Backing"})

if has_cmyk:
    cmyk_results = rgb_to_cmyk_coverage(img_rgb, cmyk_area, cmyk_method)
    for plate, pct in cmyk_results.items():
        if plate in ["CMYK photo area", "CMYK total laydown"]:
            result_type = "CMYK Summary"
        else:
            result_type = f"CMYK Photo Estimate ({cmyk_method})"
        results.append({"Ink / Plate": plate, "Coverage %": round(pct, 2), "Type": result_type})

df = pd.DataFrame(results)

st.subheader("Result")
st.dataframe(df, use_container_width=True)

laydown_rows = df[~df["Ink / Plate"].isin(["CMYK photo area"])]
st.success(f"Total ink laydown estimate: {laydown_rows['Coverage %'].sum():.2f}%")

if has_cmyk:
    st.caption(
        "CMYK breakdown is estimated from the uploaded image. "
        "GCR/UCR is more realistic for photos because neutral gray/dark areas move into K black. "
        "Exact CMYK requires the original AI/PDF separation file."
    )

# =========================================================
# RESULT PREVIEWS
# =========================================================
transparent_full = cv2.resize(transparent_mask.astype(np.uint8), (img_rgb_original.shape[1], img_rgb_original.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

preview_cols = st.columns(3)

if has_cmyk:
    cmyk_full = cv2.resize(cmyk_area.astype(np.uint8), (img_rgb_original.shape[1], img_rgb_original.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    with preview_cols[0]:
        st.image(overlay_mask_preview(img_rgb_original, cmyk_full, transparent_full), caption="Final CMYK/photo area", use_container_width=True)

if has_spot:
    spot_full = cv2.resize(combined_spot_mask.astype(np.uint8), (img_rgb_original.shape[1], img_rgb_original.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    with preview_cols[1]:
        st.image(overlay_mask_preview(img_rgb_original, spot_full, transparent_full), caption="Final spot ink area", use_container_width=True)

white_full = cv2.resize(white_plate_mask.astype(np.uint8), (img_rgb_original.shape[1], img_rgb_original.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
with preview_cols[2]:
    white_preview = np.full_like(img_rgb_original, 225)
    white_preview[white_full] = [255, 255, 255]
    white_preview[transparent_full] = [175, 175, 175]
    st.image(white_preview, caption="Final white plate", use_container_width=True)

st.caption("This is an image-based quotation estimate. For exact production separation %, use the original AI/PDF separation file.")
