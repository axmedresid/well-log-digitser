# MIT License
#
# Copyright (c) 2026
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Computer Vision-Based Well Log Digitization
Raster well-log image (PNG/JPG/TIFF/BMP) -> LAS 2.0 + QC image.

Dependencies:  pip install opencv-python numpy scipy lasio
(tkinter ships with the standard Python installer.)

Known limitations:
  * Wrap-around / backup scales are not supported.
  * The tracker can be pulled toward text, labels or a second curve that sits
    very close to the target curve; always check the QC image.
"""

import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import cv2
import lasio
import numpy as np
from scipy.signal import savgol_filter

NULL_VALUE = -999.25

# HSV hue bands (OpenCV hue range is 0-179) and the BGR channel index that
# dominates for each supported curve colour.
COLOR_PRESETS = {
    "red": {"hue_ranges": [(0, 12), (165, 180)], "channel": 2},
    "green": {"hue_ranges": [(40, 85)], "channel": 1},
    "blue": {"hue_ranges": [(95, 130)], "channel": 0},
}


# ---------------------------------------------------------
# Unicode-safe image I/O (cv2.imread/imwrite fail on non-ASCII Windows paths)
# ---------------------------------------------------------
def read_image(path: str) -> np.ndarray:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read file: {path}") from exc
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img


def write_image(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Cannot encode image: {path}")
    buf.tofile(path)


# ---------------------------------------------------------
# 0. Native File Picker Dialog (tkinter)
# ---------------------------------------------------------
def select_image_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select Raster Well-Log Image",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
            ("All Files", "*.*"),
        ],
    )
    root.destroy()

    if not file_path:
        raise SystemExit("No file selected. Exiting script.")

    return file_path


# ---------------------------------------------------------
# 1. Petrophysical Track & Mode Configuration Dialog
# ---------------------------------------------------------
def get_petrophysical_parameters() -> dict:
    params = {}
    root = tk.Tk()
    root.title("Petrophysical Track & Mode Calibration")
    root.attributes("-topmost", True)
    root.resizable(False, False)
    pad = {"padx": 12, "pady": 5}

    row = 0
    tk.Label(root, text="Curve Scan Mode:", anchor="w", font=("Arial", 9, "bold")).grid(row=row, column=0, sticky="w", **pad)
    mode_var = tk.StringVar(value="color")
    mode_frame = tk.Frame(root)
    mode_frame.grid(row=row, column=1, sticky="w", **pad)
    tk.Radiobutton(mode_frame, text="Color curve", variable=mode_var, value="color").pack(anchor="w")
    tk.Radiobutton(mode_frame, text="B&W / Grayscale Scan", variable=mode_var, value="grayscale").pack(anchor="w")

    row += 1
    tk.Label(root, text="Curve Colour (Color mode):", anchor="w").grid(row=row, column=0, sticky="w", **pad)
    color_var = tk.StringVar(value="red")
    color_frame = tk.Frame(root)
    color_frame.grid(row=row, column=1, sticky="w", **pad)
    for name in COLOR_PRESETS:
        tk.Radiobutton(color_frame, text=name.capitalize(), variable=color_var, value=name).pack(side="left")

    fields = [
        ("Well Name:", "well_name", "WELL-1"),
        ("Curve Mnemonic:", "mnemonic", "RHOB"),
        ("Curve Unit:", "unit", "G/CC"),
        ("Left Scale Value:", "left_scale", "1.95"),
        ("Right Scale Value:", "right_scale", "2.95"),
    ]
    vars_ = {}
    for label, key, default in fields:
        row += 1
        tk.Label(root, text=label, anchor="w").grid(row=row, column=0, sticky="w", **pad)
        vars_[key] = tk.StringVar(value=default)
        tk.Entry(root, textvariable=vars_[key], width=18).grid(row=row, column=1, sticky="w", **pad)

    row += 1
    tk.Label(root, text="Track Scale Type:", anchor="w").grid(row=row, column=0, sticky="w", **pad)
    scale_type_var = tk.StringVar(value="linear")
    scale_frame = tk.Frame(root)
    scale_frame.grid(row=row, column=1, sticky="w", **pad)
    tk.Radiobutton(scale_frame, text="Linear", variable=scale_type_var, value="linear").pack(side="left")
    tk.Radiobutton(scale_frame, text="Log", variable=scale_type_var, value="log").pack(side="left")

    for label, key, default in [("Depth Unit (FT / M):", "depth_unit", "FT"), ("Sampling Step:", "step", "0.5")]:
        row += 1
        tk.Label(root, text=label, anchor="w").grid(row=row, column=0, sticky="w", **pad)
        vars_[key] = tk.StringVar(value=default)
        tk.Entry(root, textvariable=vars_[key], width=18).grid(row=row, column=1, sticky="w", **pad)

    def on_submit():
        try:
            l_val = float(vars_["left_scale"].get().strip())
            r_val = float(vars_["right_scale"].get().strip())
            s_val = float(vars_["step"].get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Left Scale, Right Scale, and Step must be valid numbers.")
            return

        sc_type = scale_type_var.get().strip().lower()
        if l_val == r_val:
            messagebox.showerror("Invalid Scale", "Left and Right scale values must differ.")
            return
        if sc_type == "log" and (l_val <= 0 or r_val <= 0):
            messagebox.showerror("Invalid Scale", "Logarithmic scales require positive Left and Right values (> 0).")
            return
        if s_val <= 0:
            messagebox.showerror("Invalid Step", "Sampling step must be greater than 0.")
            return

        params["curve_mode"] = mode_var.get().strip().lower()
        params["curve_color"] = color_var.get().strip().lower()
        params["well_name"] = vars_["well_name"].get().strip() or "UNKNOWN"
        params["mnemonic"] = vars_["mnemonic"].get().strip().upper() or "CURVE"
        params["unit"] = vars_["unit"].get().strip().upper() or "UNIT"
        params["left_scale"] = l_val
        params["right_scale"] = r_val
        params["scale_type"] = sc_type
        params["depth_unit"] = vars_["depth_unit"].get().strip().upper() or "FT"
        params["step"] = s_val
        root.destroy()

    row += 1
    tk.Button(
        root,
        text="Confirm & Start 4-Click Calibration",
        command=on_submit,
        bg="#1f6aa5",
        fg="white",
        font=("Arial", 10, "bold"),
        pady=5,
    ).grid(row=row, column=0, columnspan=2, pady=15)

    root.mainloop()

    if not params:
        raise SystemExit("Configuration window closed. Exiting script.")

    return params


# ---------------------------------------------------------
# 2. Dynamic 4-Point Calibration + Depth Prompt
# ---------------------------------------------------------
def prompt_depth_value(title: str, prompt_text: str, default_val: float) -> float:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    val = simpledialog.askfloat(title, prompt_text, initialvalue=default_val, parent=root)
    root.destroy()
    if val is None:
        raise SystemExit("Depth entry cancelled. Exiting script.")
    return float(val)


def get_screen_size() -> tuple:
    try:
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except tk.TclError:
        return 1600, 900


def compute_display_scale(img_w: int, img_h: int, screen_w: int, screen_h: int) -> float:
    """Enlarge very narrow images, shrink large ones so the whole image fits on screen."""
    max_w = max(400, screen_w - 100)
    max_h = max(300, screen_h - 150)
    fit = min(max_w / img_w, max_h / img_h)
    if img_w < 250:
        return min(2.0, fit)
    return min(1.0, fit)


def calibrate_track_dynamic(image_path: str, depth_unit: str):
    img = read_image(image_path)

    screen_w, screen_h = get_screen_size()
    display_scale = compute_display_scale(img.shape[1], img.shape[0], screen_w, screen_h)
    vis = cv2.resize(img, (0, 0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)

    step_prompts = [
        "1/4: Click INNER LEFT track border",
        "2/4: Click INNER RIGHT track border",
        "3/4: Click TOP depth tick mark",
        "4/4: Click BOTTOM depth tick mark",
    ]
    clicks = []
    pending_depth_prompt = [None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4 and pending_depth_prompt[0] is None:
            orig_x = min(img.shape[1] - 1, int(round(x / display_scale)))
            orig_y = min(img.shape[0] - 1, int(round(y / display_scale)))
            clicks.append((orig_x, orig_y))

            cv2.circle(vis, (x, y), 5, (255, 0, 0), -1)
            if len(clicks) in (1, 2):
                cv2.line(vis, (x, 0), (x, vis.shape[0]), (255, 0, 0), 1)
            if len(clicks) in (3, 4):
                cv2.line(vis, (0, y), (vis.shape[1], y), (255, 0, 0), 1)
                pending_depth_prompt[0] = len(clicks)

    window_name = "Dynamic Track Calibration (Esc = cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    top_depth = None
    bottom_depth = None

    while len(clicks) < 4 or pending_depth_prompt[0] is not None:
        temp = vis.copy()
        idx = min(len(clicks), 3)
        cv2.putText(temp, step_prompts[idx], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 0), 2)
        cv2.imshow(window_name, temp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Calibration cancelled by user.")
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            raise SystemExit("Calibration window closed. Exiting script.")

        if pending_depth_prompt[0] == 3:
            top_depth = prompt_depth_value(
                "Top Depth Anchor", f"Enter Top Depth ({depth_unit}) at Click 3:", default_val=11750.0
            )
            pending_depth_prompt[0] = None
        elif pending_depth_prompt[0] == 4:
            default_bot = (top_depth + 100.0) if top_depth is not None else 11850.0
            bottom_depth = prompt_depth_value(
                "Bottom Depth Anchor", f"Enter Bottom Depth ({depth_unit}) at Click 4:", default_val=default_bot
            )
            pending_depth_prompt[0] = None

    cv2.destroyAllWindows()

    y_click_3, y_click_4 = clicks[2][1], clicks[3][1]
    if y_click_3 <= y_click_4:
        y_top, y_bottom = y_click_3, y_click_4
    else:
        y_top, y_bottom = y_click_4, y_click_3
        top_depth, bottom_depth = bottom_depth, top_depth

    x_left = min(clicks[0][0], clicks[1][0])
    x_right = max(clicks[0][0], clicks[1][0])

    validate_calibration(x_left, x_right, y_top, y_bottom, top_depth, bottom_depth)
    return img, x_left, x_right, y_top, y_bottom, top_depth, bottom_depth


def validate_calibration(x_left, x_right, y_top, y_bottom, top_depth, bottom_depth):
    if x_right - x_left < 5:
        raise ValueError("Left and right track borders are too close. Please re-calibrate.")
    if y_bottom - y_top < 5:
        raise ValueError("Top and bottom depth ticks are too close. Please re-calibrate.")
    if not bottom_depth > top_depth:
        raise ValueError(
            f"Depth must increase downward: upper tick = {top_depth}, lower tick = {bottom_depth}. "
            "Check the depth values you entered."
        )


# ---------------------------------------------------------
# 3. Curve Extraction (Color vs B&W Grayscale + Grid Removal)
# ---------------------------------------------------------
def build_color_mask(roi_bgr: np.ndarray, color: str):
    if color not in COLOR_PRESETS:
        raise ValueError(f"Unsupported curve colour: {color}")
    preset = COLOR_PRESETS[color]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    for h_lo, h_hi in preset["hue_ranges"]:
        mask |= cv2.inRange(roi_hsv, np.array([h_lo, 90, 70]), np.array([h_hi, 255, 255]))

    chans = cv2.split(roi_bgr.astype(np.float32))
    main = chans[preset["channel"]]
    others = [c for i, c in enumerate(chans) if i != preset["channel"]]
    signal = main - np.maximum(others[0], others[1])
    return mask, signal


def build_grayscale_mask(roi_bgr: np.ndarray, grid_fraction: float = 0.5):
    """Binarise the dark ink, then remove grid lines.

    A column (or row) is treated as a grid line only if ink covers more than
    `grid_fraction` of its length. A real curve that runs straight for a while
    stays far below that, so it is no longer erased like it was with a short
    morphological kernel.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ink = binary > 0
    col_frac = ink.mean(axis=0)
    row_frac = ink.mean(axis=1)
    grid_cols = np.where(col_frac > grid_fraction)[0]
    grid_rows = np.where(row_frac > grid_fraction)[0]

    mask = binary.copy()
    mask[:, grid_cols] = 0
    mask[grid_rows, :] = 0
    # Drop isolated specks left over from removed lines / scan noise.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask, inv.astype(np.float32)


def fill_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of length <= max_gap; leave longer runs as NaN."""
    out = values.copy()
    isnan = np.isnan(out)
    if not isnan.any():
        return out
    idx = np.arange(len(out))
    valid_idx = idx[~isnan]
    filled = np.interp(idx, valid_idx, out[~isnan])

    run_start = None
    for i in range(len(out) + 1):
        nan_here = i < len(out) and isnan[i]
        if nan_here and run_start is None:
            run_start = i
        elif not nan_here and run_start is not None:
            run_len = i - run_start
            interior = run_start > 0 and i < len(out)
            if interior and run_len <= max_gap:
                out[run_start:i] = filled[run_start:i]
            run_start = None
    return out


def smooth_segments(values: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing applied separately to each continuous (non-NaN) segment."""
    out = values.copy()
    valid = ~np.isnan(values)
    edges = np.diff(np.concatenate(([0], valid.astype(int), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for s, e in zip(starts, ends):
        n = e - s
        win = min(window, n if n % 2 == 1 else n - 1)
        if win > polyorder + 2:
            out[s:e] = savgol_filter(values[s:e], window_length=win, polyorder=polyorder)
    return out


def track_curve(mask: np.ndarray, signal: np.ndarray) -> np.ndarray:
    roi_h, roi_w = mask.shape
    raw_x = np.full(roi_h, np.nan)
    prev_x = None
    max_jump = max(12.0, roi_w * 0.15)

    for row in range(roi_h):
        cols = np.where(mask[row, :] > 0)[0]
        if len(cols) == 0:
            continue
        if prev_x is not None:
            near = cols[np.abs(cols - prev_x) <= max_jump]
            if len(near) > 0:
                cols = near
        weights = np.maximum(signal[row, cols], 1.0)
        cx = float(np.average(cols, weights=weights))
        raw_x[row] = cx
        prev_x = cx
    return raw_x


def pixels_to_values(u: np.ndarray, params: dict) -> np.ndarray:
    left_scale = params["left_scale"]
    right_scale = params["right_scale"]
    scale_type = params["scale_type"]
    if scale_type == "linear":
        return left_scale + u * (right_scale - left_scale)
    if scale_type in ("log", "logarithmic"):
        log_min = np.log10(left_scale)
        log_max = np.log10(right_scale)
        return 10 ** (log_min + u * (log_max - log_min))
    raise ValueError(f"Unsupported scale_type: {scale_type}")


def extract_log_curve(
    img_bgr: np.ndarray,
    x1: int,
    x2: int,
    y1: int,
    y2: int,
    top_depth: float,
    bottom_depth: float,
    params: dict,
    qc_out_path: str,
    max_gap_px: int = 10,
):
    validate_calibration(x1, x2, y1, y2, top_depth, bottom_depth)

    # Inclusive crop: x2 / y2 are the clicked border / tick pixels themselves.
    roi_bgr = img_bgr[y1 : y2 + 1, x1 : x2 + 1]
    roi_h, roi_w = roi_bgr.shape[:2]

    curve_mode = params.get("curve_mode", "color")
    if curve_mode == "color":
        mask, signal = build_color_mask(roi_bgr, params.get("curve_color", "red"))
    else:
        mask, signal = build_grayscale_mask(roi_bgr)

    raw_x = track_curve(mask, signal)
    if np.sum(~np.isnan(raw_x)) < 5:
        raise RuntimeError("Insufficient curve pixels detected. Check Curve Mode / Colour or crop bounds.")

    x_filled = fill_short_gaps(raw_x, max_gap=max_gap_px)
    x_smooth = smooth_segments(x_filled)

    u = np.clip(x_smooth / float(roi_w - 1), 0.0, 1.0)
    raw_values = pixels_to_values(u, params)

    # Row 0 -> top_depth, row roi_h-1 -> bottom_depth (both are the clicked ticks).
    raw_depths = top_depth + np.arange(roi_h) * (bottom_depth - top_depth) / (roi_h - 1)

    step = params["step"]
    n_samples = int(np.floor((bottom_depth - top_depth) / step + 1e-9)) + 1
    target_depths = np.round(top_depth + step * np.arange(n_samples), 6)

    # Resample; a target depth is NULL if either neighbouring raw sample is NULL.
    valid = ~np.isnan(raw_values)
    target_values = np.interp(target_depths, raw_depths, np.where(valid, raw_values, 0.0))
    near_invalid = np.interp(target_depths, raw_depths, (~valid).astype(float)) > 0
    target_values[near_invalid] = np.nan

    qc_img = roi_bgr.copy()
    trace_color = (0, 0, 255) if curve_mode == "grayscale" else (255, 0, 0)
    if params.get("curve_color") == "blue" and curve_mode == "color":
        trace_color = (0, 0, 255)
    for r_idx, x_val in enumerate(x_smooth):
        if not np.isnan(x_val):
            cv2.circle(qc_img, (int(round(x_val)), r_idx), 1, trace_color, -1)
    write_image(qc_out_path, qc_img)

    return target_depths, target_values


# ---------------------------------------------------------
# 4. Export LAS 2.0
# ---------------------------------------------------------
def save_las2(depths: np.ndarray, values: np.ndarray, params: dict, filename: str):
    depth_unit = params["depth_unit"]
    las = lasio.LASFile()
    las.well.STRT.value = float(depths[0])
    las.well.STOP.value = float(depths[-1])
    las.well.STEP.value = float(params["step"])
    for key in ("STRT", "STOP", "STEP"):
        las.well[key].unit = depth_unit
    las.well.NULL.value = NULL_VALUE
    las.well.WELL.value = params.get("well_name", "UNKNOWN")

    out_values = np.where(np.isnan(values), NULL_VALUE, np.round(values, 4))

    las.append_curve("DEPT", depths, unit=depth_unit, descr="Measured Depth")
    las.append_curve(
        params["mnemonic"],
        out_values,
        unit=params["unit"],
        descr=f"Digitized {params['mnemonic']} ({params['scale_type'].upper()} scale)",
    )
    with open(filename, "w", encoding="utf-8") as fh:
        las.write(fh, version=2.0)


def fmt_depth(d: float) -> str:
    return f"{d:g}".replace(".", "p")


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    selected_image = select_image_file()
    print(f"Selected file: {selected_image}")

    petro_params = get_petrophysical_parameters()
    print(f"Configured Parameters: {petro_params}")

    try:
        img, x1, x2, y1, y2, top_d, bot_d = calibrate_track_dynamic(
            selected_image, depth_unit=petro_params["depth_unit"]
        )
    except ValueError as exc:
        messagebox.showerror("Calibration Error", str(exc))
        raise SystemExit(str(exc))
    print(f"Calibrated Interval: {top_d} to {bot_d} {petro_params['depth_unit']}")

    folder = os.path.dirname(selected_image)
    base_name = os.path.splitext(os.path.basename(selected_image))[0]
    mnemonic = petro_params["mnemonic"]
    d_tag = f"{fmt_depth(top_d)}_{fmt_depth(bot_d)}"

    las_output_path = os.path.join(folder, f"{base_name}_{mnemonic}_{d_tag}.las")
    qc_output_path = os.path.join(folder, f"{base_name}_{mnemonic}_{d_tag}_QC.png")

    try:
        depths, curve_vals = extract_log_curve(
            img, x1, x2, y1, y2, top_d, bot_d, petro_params, qc_out_path=qc_output_path
        )
    except (RuntimeError, ValueError) as exc:
        messagebox.showerror("Extraction Error", str(exc))
        raise SystemExit(str(exc))

    save_las2(depths, curve_vals, petro_params, las_output_path)

    n_null = int(np.isnan(curve_vals).sum())
    print(f"\n[DONE] Saved LAS 2.0: {las_output_path}  ({len(depths)} samples, {n_null} NULL)")
    print(f"[DONE] Saved QC image: {qc_output_path}")


if __name__ == "__main__":
    main()
