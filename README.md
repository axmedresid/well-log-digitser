Computer Vision-Based Well Log Digitization: extracts a curve from a raster
well-log image (PNG/JPG/TIFF) and saves it as a LAS 2.0 file plus a QC image.

## Install
pip install -r requirements.txt

## Run
python digitize_log_dynamic_cv.py

## Steps
1. Pick the log image.
2. Enter curve settings (mnemonic, unit, scale values, linear/log, step).
3. Click left and right track borders, then top and bottom depth ticks and type their depths.
4. Check the QC image; the LAS file is saved next to the input image.

## Limitations
- Wrap-around (backup) scales are not supported.
- Text or a second curve close to the target curve can mislead the tracker. Check the QC image.
