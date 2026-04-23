#!/usr/bin/env python3
"""
hale_coronagraph_fetch.py
HALE Season 2 — Coronagraph change detection pipeline
Enneagrid Research Consortium · heliodata.ai

Instruments: STEREO-A COR2 + SOHO LASCO C3
Cadence: every 30 min (cron)
Output: /var/www/heliodata.ai/html/data/hale_coronagraph.jsonl
Frames: /var/www/heliodata.ai/html/data/coronagraph/ (rolling window, max 10 pairs)

Change detection: Wolfram Engine (wolframscript)
  - Masked brightness computation (comet quadrant only)
  - ImageDifference tail proxy (streak detection)
  - UUID via CreateUUID[]
Solar contamination: GOES X-ray cross-reference
  - If GOES flux >= 1e-5 (C-class+) at computation time -> solar_contamination_flag
"""

import os
import sys
import json
import time
import shutil
import logging
import requests
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
BASE        = Path('/var/www/heliodata.ai/html/data')
FRAME_DIR   = BASE / 'coronagraph'
JSONL_PATH  = BASE / 'hale_coronagraph.jsonl'
LOG_PATH    = Path('/var/log/hale_coronagraph.log')
MAX_PAIRS   = 5   # keep 5 pairs = 10 JPEGs max per instrument

# ── Sources ───────────────────────────────────────────────────────────────
SOURCES = {
    'cor2': {
        'url':   'https://stereo-ssc.nascom.nasa.gov/beacon/latest/ahead_cor2_latest.jpg',
        'label': 'STEREO-A COR2',
        # Comet quadrant mask: left half, vertical centre band
        # COR2 field: comet seen lower-left of occulter in Apr 22 frame
        'mask_region': 'left',
    },
    'c3': {
        'url':   'https://soho.nascom.nasa.gov/data/realtime/c3/512/latest.jpg',
        'label': 'SOHO LASCO C3',
        # C3 field: comet approaches from lower quadrant toward conjunction
        'mask_region': 'lower_left',
    },
}

GOES_URL = 'https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json'

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
log = logging.getLogger('hale_coronagraph')

# ── GOES X-ray check ──────────────────────────────────────────────────────
def get_goes_flux():
    """Return latest GOES 1-8A flux. Returns None on failure."""
    try:
        r = requests.get(GOES_URL, timeout=15)
        if not r.ok:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        # Filter 1-8 Angstrom long channel
        long = [x for x in data if '1-8' in str(x.get('energy','')) or
                'long' in str(x.get('band','')).lower()]
        series = long if long else data
        latest = series[-1]
        flux = float(latest.get('flux') or latest.get('observed_flux') or
                     latest.get('value') or 0)
        return flux
    except Exception as e:
        log.warning(f"GOES fetch failed: {e}")
        return None

def goes_contamination_flag(flux):
    """C-class or above (>=1e-5) flags potential solar contamination."""
    if flux is None:
        return False, 'unknown'
    if flux >= 1e-3:
        return True, 'X'
    elif flux >= 1e-4:
        return True, 'M'
    elif flux >= 1e-5:
        return True, 'C'
    else:
        return False, f'B{flux/1e-6:.1f}' if flux >= 1e-6 else 'A'


# ── COMET CENTROID DETECTION ──────────────────────────────────────────────
def detect_comet_centroid(image_path, search_rows=(200,450), search_cols=(20,250),
                           occulter_centre=(256,256), occulter_radius=110):
    """
    Detects brightest compact source in COR2 image — the comet.
    Uses weighted centroid with compactness filter to distinguish comet
    from diffuse coronal streamers (remote sensing anomaly detection approach).

    Returns dict with centroid_x, centroid_y, coma_radius_px, peak_brightness,
    compactness_ratio. Returns None if no compact source found.
    """
    try:
        from PIL import Image, ImageDraw
        import numpy as np

        img = Image.open(image_path)
        arr = np.array(img)
        h, w = arr.shape[:2]
        gray = np.mean(arr, axis=2)

        # Mask occulter disk
        Y, X = np.ogrid[:h, :w]
        cy_o, cx_o = occulter_centre
        occulter_mask = (Y - cy_o)**2 + (X - cx_o)**2 < occulter_radius**2
        masked = gray.copy()
        masked[occulter_mask] = 0

        # Extract search ROI
        r0, r1 = search_rows
        c0, c1 = search_cols
        roi = masked[r0:r1, c0:c1]

        if roi.max() < 50:
            log.warning("detect_comet_centroid: ROI too dim, no source found")
            return None

        # Find brightest pixel in ROI
        flat = roi.flatten()
        best_flat = int(np.argmax(flat))
        best_r = best_flat // roi.shape[1] + r0
        best_c = best_flat % roi.shape[1] + c0

        # Compactness check — reject diffuse streamers
        margin = 15
        neigh = gray[max(0,best_r-margin):best_r+margin,
                     max(0,best_c-margin):best_c+margin]
        peak_val = float(gray[best_r, best_c])
        compactness = peak_val / (neigh.mean() + 1e-6)
        log.info(f"Centroid candidate ({best_r},{best_c}) brightness={peak_val:.1f} "
                 f"compactness={compactness:.2f}")

        if compactness < 2.5:
            log.warning(f"detect_comet_centroid: compactness {compactness:.2f} < 2.5, "
                        "likely streamer not comet")
            return None

        # Refine centroid with weighted mean (squared weights emphasise peak)
        reg = gray[best_r-margin:best_r+margin, best_c-margin:best_c+margin]
        weights = reg ** 2
        total = weights.sum()
        ry = np.arange(best_r-margin, best_r+margin)
        rx = np.arange(best_c-margin, best_c+margin)
        refined_y = int((weights.sum(axis=1) * ry).sum() / total)
        refined_x = int((weights.sum(axis=0) * rx).sum() / total)

        # Dynamic coma radius from brightness falloff
        peak = gray[refined_y, refined_x]
        threshold = peak * 0.15
        radius = 8
        for r in range(8, 45):
            r0b = max(0, refined_y-r); r1b = min(h, refined_y+r)
            c0b = max(0, refined_x-r); c1b = min(w, refined_x+r)
            ring_mean = gray[r0b:r1b, c0b:c1b].mean()
            if ring_mean < threshold:
                radius = r + 6
                break

        log.info(f"Refined centroid: ({refined_y},{refined_x}) "
                 f"coma_radius={radius}px compactness={compactness:.2f}")

        return {
            'centroid_y': refined_y,
            'centroid_x': refined_x,
            'coma_radius_px': radius,
            'peak_brightness': round(peak_val, 3),
            'compactness_ratio': round(compactness, 3),
        }

    except Exception as e:
        log.error(f"detect_comet_centroid: {e}")
        return None


def annotate_cor2_frame(image_path, output_path, detection):
    """
    Draws double circle + crosshair on COR2 frame at detected comet position.
    Saves to output_path. Never modifies the original.
    """
    try:
        from PIL import Image, ImageDraw
        img = Image.open(image_path).copy()
        draw = ImageDraw.Draw(img)
        cy = detection['centroid_y']
        cx = detection['centroid_x']
        r  = detection['coma_radius_px']

        # Outer circle — region of interest boundary
        draw.ellipse([cx-r-5, cy-r-5, cx+r+5, cy+r+5],
                     outline=(0, 255, 200), width=1)
        # Inner circle — coma boundary
        draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                     outline=(0, 200, 160), width=1)
        # Crosshair
        gap = 3
        arm = 7
        draw.line([(cx-arm-gap, cy), (cx-gap, cy)], fill=(0,255,200), width=1)
        draw.line([(cx+gap, cy), (cx+arm+gap, cy)], fill=(0,255,200), width=1)
        draw.line([(cx, cy-arm-gap), (cx, cy-gap)], fill=(0,255,200), width=1)
        draw.line([(cx, cy+gap), (cx, cy+arm+gap)], fill=(0,255,200), width=1)

        img.save(output_path, quality=92)
        # Fix permissions
        os.chmod(output_path, 0o644)
        try:
            import pwd, grp
            uid = pwd.getpwnam('www-data').pw_uid
            gid = grp.getgrnam('www-data').gr_gid
            os.chown(output_path, uid, gid)
        except Exception:
            pass
        log.info(f"Annotated frame saved: {output_path}")
        return True
    except Exception as e:
        log.error(f"annotate_cor2_frame: {e}")
        return False


def wolfram_centroid_uuid(detection):
    """
    Stamps centroid detection with Wolfram CAG UUID.
    Returns uuid string or None.
    """
    wl = f"""
Module[{{uuid}},
  uuid = ToString[CreateUUID[]];
  ExportString[{{
    "centroid_x"       -> {detection['centroid_x']},
    "centroid_y"       -> {detection['centroid_y']},
    "coma_radius_px"   -> {detection['coma_radius_px']},
    "peak_brightness"  -> {detection['peak_brightness']},
    "compactness"      -> {detection['compactness_ratio']},
    "method"           -> "weighted_centroid_v1",
    "cag_uuid"         -> uuid
  }}, "JSON"]
]
"""
    try:
        result = subprocess.run(
            ['wolframscript', '-code', wl],
            capture_output=True, text=True, timeout=90
        )
        if result.returncode != 0:
            log.error(f"wolfram_centroid_uuid: {result.stderr[:150]}")
            return None
        parsed = json.loads(result.stdout.strip())
        if isinstance(parsed, list):
            d = {item[0]: item[1] for item in parsed if isinstance(item, list)}
            return d.get('cag_uuid')
        return parsed.get('cag_uuid')
    except Exception as e:
        log.error(f"wolfram_centroid_uuid: {e}")
        return None

# ── Frame fetch ───────────────────────────────────────────────────────────
def fetch_frame(instrument, source):
    """
    Download latest frame. Rotate current->previous. Return (current_path, previous_path).
    Returns (None, None) on failure.
    """
    cur  = FRAME_DIR / f'{instrument}_current.jpg'
    prev = FRAME_DIR / f'{instrument}_previous.jpg'

    try:
        r = requests.get(source['url'], timeout=30, stream=True)
        if not r.ok:
            log.error(f"{instrument}: HTTP {r.status_code}")
            return None, None

        # Validate it's actually an image before rotating files
        raw = r.content
        if len(raw) < 1000:
            log.error(f"{instrument}: response too small ({len(raw)} bytes) — bad beacon frame")
            return None, None

        # Quick PIL validation
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            img = Image.open(tmp_path)
            img.verify()
        except Exception as e:
            log.error(f"{instrument}: image validation failed — {e}")
            os.unlink(tmp_path)
            return None, None

        # Rotate: current -> previous
        if cur.exists():
            shutil.copy2(cur, prev)

        # Save new current
        shutil.move(tmp_path, cur)
        # Fix permissions immediately so PIL can read it
        os.chmod(cur, 0o644)
        try:
            import pwd, grp
            os.chown(cur, pwd.getpwnam('www-data').pw_uid, grp.getgrnam('www-data').gr_gid)
        except Exception:
            pass
        log.info(f"{instrument}: frame saved ({len(raw)/1024:.1f} KB)")
        return cur, prev if prev.exists() else None

    except Exception as e:
        log.error(f"{instrument}: fetch exception — {e}")
        return None, None

# ── Wolfram change detection ──────────────────────────────────────────────
def wolfram_change_detection(instrument, current_path, previous_path, mask_region):
    """
    Run Wolfram Engine change detection between two frames.
    Returns dict with brightness values, delta, tail proxy, UUID.
    Returns None on failure.

    Mask regions:
      'left'       — left half of image (COR2 comet quadrant)
      'lower_left' — lower-left quadrant (C3 comet approach zone)
      'full'       — full image (fallback)
    """
    # Build Wolfram expression
    # Mask logic: extract subimage before computing to reduce solar disk contamination
    wl_mask = {
        'left':       'ImageTake[#, All, {1, Floor[ImageDimensions[#][[1]]/2]}]&',
        'lower_left': 'ImageTake[#, {Floor[ImageDimensions[#][[2]]/2], -1}, {1, Floor[ImageDimensions[#][[1]]/2]}]&',
        'full':       'Identity',
    }.get(mask_region, 'Identity')

    wl_expr = f"""
Module[
  {{cur, prev, curM, prevM, diff, brightCur, brightPrev, delta, tailProxy, uuid}},
  cur  = Import["{current_path}"];
  prev = Import["{previous_path}"];
  (* Apply mask to focus on comet quadrant *)
  curM  = ({wl_mask})[cur];
  prevM = ({wl_mask})[prev];
  (* Mean brightness of masked region, 0-1 scale *)
  brightCur  = Mean[Flatten[ImageData[ColorConvert[curM,  "Grayscale"]]]];
  brightPrev = Mean[Flatten[ImageData[ColorConvert[prevM, "Grayscale"]]]];
  delta = brightCur - brightPrev;
  (* Tail proxy: std dev of difference image — spikes when streak feature moves *)
  diff = ImageDifference[curM, prevM];
  tailProxy = StandardDeviation[Flatten[ImageData[ColorConvert[diff, "Grayscale"]]]];
  uuid = ToString[CreateUUID[]];
  ExportString[
    {{
      "brightness_current"  -> N[brightCur,  6],
      "brightness_previous" -> N[brightPrev, 6],
      "brightness_delta"    -> N[delta,       6],
      "tail_proxy"          -> N[tailProxy,   6],
      "mask_region"         -> "{mask_region}",
      "cag_uuid"            -> uuid
    }},
    "JSON"
  ]
]
"""

    try:
        result = subprocess.run(
            ['wolframscript', '-code', wl_expr],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            log.error(f"{instrument}: wolframscript error — {result.stderr[:200]}")
            return None

        output = result.stdout.strip()
        # Parse JSON output from Wolfram
        # Wolfram ExportString JSON uses -> which is valid JSON with quotes
        parsed = json.loads(output)

        # Wolfram returns list of [key, value] pairs from Association
        if isinstance(parsed, list):
            d = {}
            for item in parsed:
                if isinstance(item, list) and len(item) == 2:
                    d[item[0]] = item[1]
            return d
        elif isinstance(parsed, dict):
            return parsed
        else:
            log.error(f"{instrument}: unexpected Wolfram output format")
            return None

    except subprocess.TimeoutExpired:
        log.error(f"{instrument}: wolframscript timeout (120s)")
        return None
    except json.JSONDecodeError as e:
        log.error(f"{instrument}: JSON parse error — {e} — output: {result.stdout[:200]}")
        return None
    except Exception as e:
        log.error(f"{instrument}: Wolfram exception — {e}")
        return None

# ── Rolling window cleanup ────────────────────────────────────────────────
def cleanup_rolling_window():
    """Keep only current + previous per instrument. Remove any older numbered frames."""
    kept = 0
    for f in sorted(FRAME_DIR.glob('*.jpg')):
        name = f.stem
        # Keep: cor2_current, cor2_previous, c3_current, c3_previous
        if any(name.endswith(s) for s in ['_current', '_previous', '_annotated']):
            kept += 1
        else:
            log.info(f"Rolling window: removing {f.name}")
            f.unlink()

    log.info(f"Rolling window: {kept} frames retained")

# ── JSONL entry writer ────────────────────────────────────────────────────
def write_jsonl_entry(entry):
    with open(JSONL_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    # Keep JSONL to last 500 entries
    try:
        lines = JSONL_PATH.read_text().splitlines()
        if len(lines) > 500:
            JSONL_PATH.write_text('\n'.join(lines[-500:]) + '\n')
    except Exception:
        pass

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    log.info("=== hale_coronagraph_fetch start ===")
    utc_now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # 1. GOES contamination check
    goes_flux = get_goes_flux()
    contaminated, flare_class = goes_contamination_flag(goes_flux)
    log.info(f"GOES flux: {goes_flux} — class: {flare_class} — contamination_flag: {contaminated}")

    results = {}
    centroid_result = None

    for instrument, source in SOURCES.items():
        log.info(f"--- {instrument.upper()} ---")

        # 2. Fetch frame
        cur_path, prev_path = fetch_frame(instrument, source)
        if cur_path is None:
            log.warning(f"{instrument}: skipping — fetch failed")
            results[instrument] = {'error': 'fetch_failed'}
            continue

        # 3. Change detection (requires both frames)
        # COR2: run centroid detection on every frame (not just when diff available)
        if instrument == 'cor2' and cur_path is not None:
            detection = detect_comet_centroid(str(cur_path))
            if detection:
                centroid_uuid = wolfram_centroid_uuid(detection)
                detection['cag_uuid'] = centroid_uuid
                centroid_result = detection
                annotated_path = FRAME_DIR / 'cor2_annotated.jpg'
                annotate_cor2_frame(str(cur_path), str(annotated_path), detection)
                log.info(f"Centroid detection: ({detection['centroid_y']},{detection['centroid_x']}) "
                         f"r={detection['coma_radius_px']}px uuid={centroid_uuid}")
            else:
                log.warning("cor2: centroid detection failed — annotated frame not updated")

        if prev_path is None:
            log.info(f"{instrument}: no previous frame yet — first run, skipping diff")
            results[instrument] = {
                'brightness_current': None,
                'brightness_delta': None,
                'tail_proxy': None,
                'cag_uuid': None,
                'note': 'first_frame_no_diff'
            }
            continue

        wl_result = wolfram_change_detection(
            instrument, str(cur_path), str(prev_path), source['mask_region']
        )

        if wl_result is None:
            log.warning(f"{instrument}: Wolfram computation failed")
            results[instrument] = {'error': 'wolfram_failed'}
            continue

        results[instrument] = wl_result
        log.info(f"{instrument}: delta={wl_result.get('brightness_delta'):.4f} "
                 f"tail_proxy={wl_result.get('tail_proxy'):.4f} "
                 f"uuid={wl_result.get('cag_uuid')}")

    # 4. Build and write JSONL entry
    entry = {
        'utc': utc_now,
        'pipeline': 'hale_coronagraph_v1',
        'goes_flux': goes_flux,
        'goes_class': flare_class,
        'solar_contamination_flag': contaminated,
        'instruments': {}
    }

    for instrument, source in SOURCES.items():
        r = results.get(instrument, {})
        entry['instruments'][instrument] = {
            'label':               source['label'],
            'mask_region':         source['mask_region'],
            'brightness_current':  r.get('brightness_current'),
            'brightness_previous': r.get('brightness_previous'),
            'brightness_delta':    r.get('brightness_delta'),
            'tail_proxy':          r.get('tail_proxy'),
            'cag_uuid':            r.get('cag_uuid'),
            'error':               r.get('error'),
            'note':                r.get('note'),
        }

    # Centroid detection result
    entry['comet_detection'] = centroid_result if centroid_result else {
        'centroid_x': None, 'centroid_y': None,
        'coma_radius_px': None, 'peak_brightness': None,
        'compactness_ratio': None, 'cag_uuid': None,
        'note': 'no_detection'
    }

    # Forward scatter flag: C3 brightness rising + low solar contamination
    c3 = entry['instruments'].get('c3', {})
    delta = c3.get('brightness_delta')
    entry['forward_scatter_candidate'] = (
        delta is not None and
        delta > 0.005 and
        not contaminated
    )

    write_jsonl_entry(entry)
    log.info(f"JSONL entry written: {utc_now} "
             f"forward_scatter={entry['forward_scatter_candidate']}")

    # 5. Rolling window cleanup
    cleanup_rolling_window()

    log.info("=== hale_coronagraph_fetch complete ===")
    print(json.dumps(entry, indent=2))

if __name__ == '__main__':
    main()
