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
        "Install once:\n"
        "pip install streamlit-image-coordinates\n\n"
        "Then run again."
    )
    st.stop()

# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="Ink Coverage Calculator", layout="wide")

MAX_PIXELS = 850_000
DISPLAY_WIDTH = 1150

st.title("🎨 Ink Coverage Calculator for Packaging")
st.caption(
    "Upload the full artwork image. Click the film/background color. "
    "Select the actual ink colors. Then calculate."
)

# =========================================================
# SESSION
# =========================================================
defaults = {
    "photo_boxes": [],
    "first_corner": None,
    "last_click": None,
    "picked_bg_hex": "#9fa1a3",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def clear_photo_boxes():
    st.session_state.photo_boxes = []
    st.session_state.first_corner = None
    st.session_state.last_click = None

# =========================================================
# HELPERS
# =========================================================
def resize_for_analysis(img_rgb):
    h, w = img_rgb.shape[:2]
    scale = min(1.0, (MAX_PIXELS / (h * w)) ** 0.5)
    if scale < 1.0:
        return cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_rgb.copy()

def hex_to_rgb(hex_color):
    hex_color = hex_color.replace("#", "")
    return np.array(
        [int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)],
        dtype=np.uint8
    )

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def rgb_to_lab_image(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

def rgb_to_lab_color(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

def color_distance_lab(img_rgb, target_rgb):
    img_lab = rgb_to_lab_image(img_rgb)
    target_lab = rgb_to_lab_color(target_rgb)
    return np.sqrt(np.sum((img_lab - target_lab) ** 2, axis=2))

def make_color_mask(img_rgb, hex_color, tolerance):
    return color_distance_lab(img_rgb, hex_to_rgb(hex_color)) <= tolerance

def make_edge_connected_background_mask(img_rgb, bg_hex, tolerance):
    candidate = make_color_mask(img_rgb, bg_hex, tolerance).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(candidate, connectivity=8)

    if num_labels <= 1:
        return candidate.astype(bool)

    edge_labels = set()
    edge_labels.update(np.unique(labels[0, :]).tolist())
    edge_labels.update(np.unique(labels[-1, :]).tolist())
    edge_labels.update(np.unique(labels[:, 0]).tolist())
    edge_labels.update(np.unique(labels[:, -1]).tolist())
    edge_labels.discard(0)

    return np.isin(labels, list(edge_labels))

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

def draw_photo_boxes(img_pil, boxes, current_point=None):
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i, (x1p, y1p, x2p, y2p) in enumerate(boxes):
        x1, x2 = sorted([int(w * x1p / 100), int(w * x2p / 100)])
        y1, y2 = sorted([int(h * y1p / 100), int(h * y2p / 100)])

        draw.rectangle([x1, y1, x2, y2], outline="blue", width=5)
        draw.rectangle([x1, max(0, y1 - 28), x1 + 115, y1], fill="white", outline="blue")
        draw.text((x1 + 6, max(0, y1 - 24)), f"PHOTO {i+1}", fill="blue")

    if current_point is not None:
        x, y = current_point
        r = 8
        draw.ellipse([x-r, y-r, x+r, y+r], fill="red", outline="white", width=2)
        draw.text((x+12, y+8), "1st corner", fill="red")

    return img

def preview_removed_area(img_rgb, mask):
    preview = img_rgb.copy()
    overlay = preview.copy()
    overlay[mask] = [255, 0, 0]
    return cv2.addWeighted(preview, 0.68, overlay, 0.32, 0)

def preview_mask_only(img_rgb, mask, background_mask=None):
    preview = np.full_like(img_rgb, 225)
    preview[mask] = img_rgb[mask]
    if background_mask is not None:
        preview[background_mask] = [175, 175, 175]
    return preview

def preview_white_plate(img_rgb, mask, background_mask=None):
    preview = np.full_like(img_rgb, 225)
    preview[mask] = [255, 255, 255]
    if background_mask is not None:
        preview[background_mask] = [175, 175, 175]
    return preview

def clean_small_noise(mask, min_area):
    u8 = mask.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)

    cleaned = np.zeros_like(u8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned > 0

def make_visible_white_mask(img_rgb, background_mask):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    return ((S <= 35) & (V >= 235)) & (~background_mask)

def make_printed_object_mask(img_rgb, background_mask, bg_hex, sensitivity):
    """
    Detect printed artwork by finding areas that are different from the clicked background/film color.
    This includes colored artwork and visible white areas.
    """
    bg_like = make_color_mask(img_rgb, bg_hex, sensitivity)
    visible_white = make_visible_white_mask(img_rgb, background_mask)

    mask = ((~bg_like) & (~background_mask)) | visible_white

    u8 = mask.astype(np.uint8) * 255
    u8 = cv2.morphologyEx(
        u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    min_area = max(25, int(img_rgb.shape[0] * img_rgb.shape[1] * 0.00020))
    return clean_small_noise(u8 > 0, min_area)

def assign_pixels_to_spot_inks(img_rgb, selected_colors, candidate_area, exclude_mask):
    img_lab = rgb_to_lab_image(img_rgb)
    names = list(selected_colors.keys())

    stack = []
    for name in names:
        target_lab = rgb_to_lab_color(hex_to_rgb(selected_colors[name]))
        stack.append(np.sqrt(np.sum((img_lab - target_lab) ** 2, axis=2)))

    nearest = np.argmin(np.stack(stack, axis=0), axis=0)
    valid = candidate_area & (~exclude_mask)

    return {name: (nearest == i) & valid for i, name in enumerate(names)}

def coverage_percent(mask, total_pixels):
    if total_pixels <= 0:
        return 0.0
    return np.sum(mask) / total_pixels * 100

def rgb_to_cmyk_coverage(img_rgb, area, total_pixels, method):
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
        C0 = 1 - R
        M0 = 1 - G
        Y0 = 1 - B

        neutral = np.minimum.reduce([C0, M0, Y0])
        max_rgb = np.maximum.reduce([R, G, B])
        min_rgb = np.minimum.reduce([R, G, B])
        chroma = max_rgb - min_rgb
        darkness = 1 - max_rgb

        gcr_strength = 0.55 + 0.35 * darkness - 0.25 * chroma
        gcr_strength = np.clip(gcr_strength, 0.35, 0.90)

        K = neutral * gcr_strength
        C = C0 - K
        M = M0 - K
        Y = Y0 - K

        ucr = np.clip((darkness - 0.45) / 0.55, 0, 1) * 0.25
        C *= (1 - ucr)
        M *= (1 - ucr)
        Y *= (1 - ucr)

        C, M, Y, K = [np.clip(x, 0, 1) for x in [C, M, Y, K]]

    C[~area], M[~area], Y[~area], K[~area] = 0, 0, 0, 0

    return {
        "CMYK Cyan": np.sum(C) / total_pixels * 100,
        "CMYK Magenta": np.sum(M) / total_pixels * 100,
        "CMYK Yellow": np.sum(Y) / total_pixels * 100,
        "CMYK Black": np.sum(K) / total_pixels * 100,
        "CMYK photo area": np.sum(area) / total_pixels * 100,
        "CMYK total laydown": (np.sum(C) + np.sum(M) + np.sum(Y) + np.sum(K)) / total_pixels * 100,
    }

# =========================================================
# 1. UPLOAD
# =========================================================
uploaded = st.file_uploader("1️⃣ Upload full artwork image", type=["png", "jpg", "jpeg"])

if not uploaded:
    st.info("Upload a full artwork image to start.")
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

total_pixels = img_rgb.shape[0] * img_rgb.shape[1]

# =========================================================
# 2. JOB TYPE
# =========================================================
st.subheader("2️⃣ Choose artwork type")

job_type = st.radio(
    "Artwork type",
    [
        "Spot colors only — no photos",
        "Spot colors + photo boxes",
        "CMYK photo only",
    ],
    index=0,
    horizontal=True,
)

has_spot = job_type in ["Spot colors only — no photos", "Spot colors + photo boxes"]
has_cmyk = job_type in ["Spot colors + photo boxes", "CMYK photo only"]

# =========================================================
# 3. BACKGROUND COLOR
# =========================================================
st.subheader("3️⃣ Click background / film color")

st.caption("Click the gray/transparent film area. Red preview = removed background / no ink.")

pick_col, control_col = st.columns([1.25, 1])

with pick_col:
    bg_click = streamlit_image_coordinates(display_img, key="bg_color_picker")

    if bg_click is not None:
        x_disp, y_disp = int(bg_click["x"]), int(bg_click["y"])

        x_orig = int(x_disp / display_w * orig_w)
        y_orig = int(y_disp / display_h * orig_h)

        x_orig = max(0, min(orig_w - 1, x_orig))
        y_orig = max(0, min(orig_h - 1, y_orig))

        picked_rgb = img_rgb_original[y_orig, x_orig]
        st.session_state.picked_bg_hex = rgb_to_hex(picked_rgb)

with control_col:
    bg_hex = st.color_picker("Selected background color", st.session_state.picked_bg_hex)

    bg_tol = st.slider(
        "Background tolerance",
        1,
        80,
        20,
        help="Higher removes more pixels close to the clicked background color."
    )

    remove_mode = st.radio(
        "Removal mode",
        ["Smart edge only - recommended", "Remove selected color everywhere"],
        index=0,
        help="Recommended protects white panels inside the artwork."
    )

if remove_mode == "Smart edge only - recommended":
    background_mask = make_edge_connected_background_mask(img_rgb, bg_hex, bg_tol)
else:
    background_mask = make_color_mask(img_rgb, bg_hex, bg_tol)

removed_pct = coverage_percent(background_mask, total_pixels)

st.markdown(f"**Removed background:** {removed_pct:.2f}%")
st.image(
    preview_removed_area(img_rgb, background_mask),
    caption="Red = removed/no ink",
    use_container_width=True
)

# =========================================================
# 4. PHOTO BOXES ONLY WHEN NEEDED
# =========================================================
cmyk_area = np.zeros(img_rgb.shape[:2], dtype=bool)

if has_cmyk:
    st.subheader("4️⃣ Select photo boxes for CMYK")
    st.caption("Only draw boxes around real photos/gradients. Do not use this for normal spot-color designs.")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 4])

    with c1:
        if st.button("Clear photo boxes"):
            clear_photo_boxes()
            st.rerun()

    with c2:
        if st.button("Undo last photo box"):
            if st.session_state.photo_boxes:
                st.session_state.photo_boxes.pop()
            st.session_state.first_corner = None
            st.rerun()

    with c3:
        if st.button("Cancel corner"):
            st.session_state.first_corner = None
            st.session_state.last_click = None
            st.rerun()

    if st.session_state.first_corner is None:
        st.info("Next click: first corner of photo box.")
    else:
        st.warning("Next click: opposite corner.")

    selector_img = draw_photo_boxes(display_img, st.session_state.photo_boxes, st.session_state.first_corner)
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

                if abs(x - x1) > 5 and abs(y - y1) > 5:
                    st.session_state.photo_boxes.append((
                        x1 / display_w * 100,
                        y1 / display_h * 100,
                        x / display_w * 100,
                        y / display_h * 100,
                    ))

                st.session_state.first_corner = None
                st.rerun()

    cmyk_area = make_rectangle_mask(img_rgb.shape, st.session_state.photo_boxes) & (~background_mask)

    st.success(f"Selected photo boxes: {len(st.session_state.photo_boxes)}")
    st.image(
        preview_mask_only(img_rgb, cmyk_area, background_mask),
        caption="CMYK/photo area preview",
        use_container_width=True
    )

# =========================================================
# 5. INK SELECTION
# =========================================================
st.subheader("5️⃣ Select actual inks")

selected_colors = {}

preset_colors = {
    "White": "#ffffff",
    "Black": "#111111",
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
        st.caption("Tick only real printed plates. Example: White, Black, Yellow.")

        cols = st.columns(4)

        for i, (name, default_hex) in enumerate(preset_colors.items()):
            with cols[i % 4]:
                checked_default = name in ["White", "Black", "Yellow"]
                checked = st.checkbox(name, value=checked_default, key=f"ink_{name}")

                if checked:
                    selected_colors[name] = st.color_picker(
                        f"{name} sample",
                        default_hex,
                        key=f"color_{name}"
                    )

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

        if n1.strip():
            selected_colors[n1.strip()] = c1v
        if n2.strip():
            selected_colors[n2.strip()] = c2v
        if n3.strip():
            selected_colors[n3.strip()] = c3v

    else:
        st.info("Spot ink selection is off for CMYK photo only mode.")

with white_col:
    st.markdown("**White ink option**")

    white_mode = st.radio(
        "How is white printed?",
        [
            "Only where white is visible",
            "Under all printed colors/photos",
        ],
        index=1,
        help=(
            "Use 'Only where white is visible' when white ink is only printed as visible white text/panels. "
            "Use 'Under all printed colors/photos' for transparent film when white backs the artwork."
        )
    )

    if has_cmyk:
        cmyk_method = st.radio(
            "CMYK breakdown method",
            ["GCR / UCR print-style CMYK", "Simple RGB-to-CMYK"],
            index=0,
        )
    else:
        cmyk_method = "GCR / UCR print-style CMYK"

# =========================================================
# 6. DETECTION PREVIEW
# =========================================================
st.subheader("6️⃣ Artwork detection preview")

st.caption(
    "Use only this slider if the preview misses artwork or includes too much background. "
    "The white plate preview updates automatically."
)

slider_col, preview_col = st.columns([1, 1.5])

with slider_col:
    object_sensitivity = st.slider(
        "Artwork detection sensitivity",
        5,
        80,
        28,
        help="Lower = stricter. Higher = includes more light/faint printed details."
    )

printed_object_mask = make_printed_object_mask(
    img_rgb,
    background_mask=background_mask,
    bg_hex=bg_hex,
    sensitivity=object_sensitivity
)

visible_white_mask = make_visible_white_mask(img_rgb, background_mask)

if white_mode == "Only where white is visible":
    live_white_mask = visible_white_mask
else:
    live_white_mask = printed_object_mask | cmyk_area

printed_pct = coverage_percent(printed_object_mask, total_pixels)
live_white_pct = coverage_percent(live_white_mask, total_pixels)

with preview_col:
    st.markdown(f"**Detected artwork:** {printed_pct:.2f}%")
    st.markdown(f"**Estimated white plate:** {live_white_pct:.2f}%")

    p1, p2 = st.columns(2)

    with p1:
        st.image(
            preview_mask_only(img_rgb, printed_object_mask, background_mask),
            caption="Detected artwork",
            use_container_width=True
        )

    with p2:
        st.image(
            preview_white_plate(img_rgb, live_white_mask, background_mask),
            caption="White plate preview",
            use_container_width=True
        )

# =========================================================
# 7. CALCULATE
# =========================================================
st.subheader("7️⃣ Calculate")

calculate = st.button("Calculate Ink Coverage", type="primary", use_container_width=True)

if not calculate:
    st.stop()

results = []
combined_spot_mask = np.zeros(img_rgb.shape[:2], dtype=bool)

if has_spot:
    if not selected_colors:
        st.error("Please select at least one spot ink.")
        st.stop()

    exclude_from_spot = cmyk_area if has_cmyk else np.zeros(img_rgb.shape[:2], dtype=bool)
    spot_candidate_area = printed_object_mask & (~exclude_from_spot)

    assigned = assign_pixels_to_spot_inks(
        img_rgb,
        selected_colors=selected_colors,
        candidate_area=spot_candidate_area,
        exclude_mask=exclude_from_spot
    )

    for ink, mask in assigned.items():
        pct = coverage_percent(mask, total_pixels)
        combined_spot_mask = combined_spot_mask | mask

        if ink == "White":
            visible_white_mask = visible_white_mask | mask

        results.append({
            "Ink / Plate": ink,
            "Coverage %": round(pct, 2),
            "Type": "Visible spot ink"
        })

if white_mode == "Only where white is visible":
    white_plate_mask = visible_white_mask
else:
    white_plate_mask = printed_object_mask | combined_spot_mask | cmyk_area

white_pct = coverage_percent(white_plate_mask, total_pixels)

results.append({
    "Ink / Plate": "White plate final",
    "Coverage %": round(white_pct, 2),
    "Type": white_mode
})

if has_cmyk:
    cmyk_results = rgb_to_cmyk_coverage(img_rgb, cmyk_area, total_pixels, cmyk_method)

    for plate, pct in cmyk_results.items():
        result_type = "CMYK Summary" if plate in ["CMYK photo area", "CMYK total laydown"] else f"CMYK Photo Estimate ({cmyk_method})"

        results.append({
            "Ink / Plate": plate,
            "Coverage %": round(pct, 2),
            "Type": result_type
        })

df = pd.DataFrame(results)

st.subheader("Result")
st.dataframe(df, use_container_width=True)

laydown_rows = df[~df["Ink / Plate"].isin(["CMYK photo area"])]
st.success(f"Total ink laydown estimate: {laydown_rows['Coverage %'].sum():.2f}%")

st.markdown(
    """
**Simple rule**

- **Only where white is visible** = white ink only for visible white text/panels.
- **Under all printed colors/photos** = white backing under the whole printed design.
"""
)

# =========================================================
# FINAL PREVIEWS
# =========================================================
st.subheader("Final previews")

background_full = cv2.resize(
    background_mask.astype(np.uint8),
    (img_rgb_original.shape[1], img_rgb_original.shape[0]),
    interpolation=cv2.INTER_NEAREST
).astype(bool)

printed_full = cv2.resize(
    printed_object_mask.astype(np.uint8),
    (img_rgb_original.shape[1], img_rgb_original.shape[0]),
    interpolation=cv2.INTER_NEAREST
).astype(bool)

spot_full = cv2.resize(
    combined_spot_mask.astype(np.uint8),
    (img_rgb_original.shape[1], img_rgb_original.shape[0]),
    interpolation=cv2.INTER_NEAREST
).astype(bool)

white_full = cv2.resize(
    white_plate_mask.astype(np.uint8),
    (img_rgb_original.shape[1], img_rgb_original.shape[0]),
    interpolation=cv2.INTER_NEAREST
).astype(bool)

cols = st.columns(3)

with cols[0]:
    st.image(
        preview_mask_only(img_rgb_original, printed_full, background_full),
        caption="Detected artwork",
        use_container_width=True
    )

with cols[1]:
    st.image(
        preview_mask_only(img_rgb_original, spot_full, background_full),
        caption="Visible spot ink area",
        use_container_width=True
    )

with cols[2]:
    st.image(
        preview_white_plate(img_rgb_original, white_full, background_full),
        caption="Final white plate",
        use_container_width=True
    )

if has_cmyk:
    cmyk_full = cv2.resize(
        cmyk_area.astype(np.uint8),
        (img_rgb_original.shape[1], img_rgb_original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    st.image(
        preview_mask_only(img_rgb_original, cmyk_full, background_full),
        caption="Final CMYK/photo area",
        use_container_width=True
    )

st.caption("Image-based quotation estimate only. Exact production percentages require original AI/PDF separations.")
