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

        # Angular position relative to Sun centre
        import math
        sun_cy, sun_cx = SUN_CENTRE['cor2']
        dx_px = refined_x - sun_cx
        dy_px = refined_y - sun_cy
        ps = PLATE_SCALE['cor2']
        angular_sep_deg = ((dx_px*ps/3600)**2 + (dy_px*ps/3600)**2)**0.5
        anti_solar_angle = math.degrees(math.atan2(-dy_px, -dx_px)) % 360

        log.info(f"Refined centroid: ({refined_y},{refined_x}) "
                 f"coma_radius={radius}px compactness={compactness:.2f} "
                 f"sep={angular_sep_deg:.3f}deg")

        return {
            'instrument':            'cor2',
            'centroid_y':            refined_y,
            'centroid_x':            refined_x,
            'coma_radius_px':        radius,
            'peak_brightness':       round(peak_val, 3),
            'compactness_ratio':     round(compactness, 3),
            'angular_sep_deg':       round(angular_sep_deg, 4),
            'anti_solar_angle':      round(anti_solar_angle, 2),
            'dx_px':                 dx_px,
            'dy_px':                 dy_px,
            'plate_scale_arcsec_px': PLATE_SCALE['cor2'],
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


# ── INSTRUMENT CALIBRATION CONSTANTS ─────────────────────────────────────
# Source: Eyles et al. 2009, Solar Physics (SECCHI instrument paper)
# All values at beacon resolution (4x or 5x binned from native)
PLATE_SCALE = {
    'cor2': 58.8,   # arcsec/pixel at 512px (native 14.7 arcsec/px × 4)
    'c3':   56.0,   # arcsec/pixel at 512px (native 11.4 arcsec/px × ~5)
    'hi2':  960.0,  # arcsec/pixel at 256px (native 4.0 arcmin/px × 4 = 16 arcmin/px)
}
# Sun centre in beacon frames (approximate, stable)
SUN_CENTRE = {
    'cor2': (256, 256),
    'c3':   (256, 256),
}
# Trajectory log path
TRAJ_PATH = BASE / 'hale_trajectory.jsonl'


# ── C3 CENTROID DETECTION ────────────────────────────────────────────────
def detect_c3_centroid(image_path):
    """
    Detects comet in LASCO C3 beacon frame.
    Search corridor: upper frame (rows 0-220) — comet entering from upper-left
    during Apr 23-27 conjunction transit.
    Uses same weighted centroid + compactness filter as COR2.
    Returns detection dict or None.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(image_path)
        arr = np.array(img)
        h, w = arr.shape[:2]
        gray = np.mean(arr, axis=2)

        # Mask C3 occulter (~65px radius at 512px)
        cy_o, cx_o = 256, 256
        Y, X = np.ogrid[:h, :w]
        occulter = (Y - cy_o)**2 + (X - cx_o)**2 < 68**2
        masked = gray.copy()
        masked[occulter] = 0

        # Search corridor — upper frame, full width
        # As conjunction approaches comet sweeps from upper-left toward Sun
        r0, r1, c0, c1 = 0, 220, 0, 512
        corridor = masked[r0:r1, c0:c1]

        if corridor.max() < 60:
            log.warning("detect_c3_centroid: corridor too dim")
            return None

        # Find brightest pixel
        flat = corridor.flatten()
        best_flat = int(np.argmax(flat))
        best_r = best_flat // corridor.shape[1] + r0
        best_c = best_flat % corridor.shape[1] + c0

        # Compactness filter
        margin = 12
        neigh = gray[max(0,best_r-margin):best_r+margin,
                     max(0,best_c-margin):best_c+margin]
        peak_val = float(gray[best_r, best_c])
        compactness = peak_val / (neigh.mean() + 1e-6)

        log.info(f"C3 centroid candidate ({best_r},{best_c}) "
                 f"brightness={peak_val:.1f} compactness={compactness:.2f}")

        if compactness < 2.5:
            log.warning(f"detect_c3_centroid: compactness {compactness:.2f} < 2.5")
            return None

        # Refined weighted centroid
        reg = gray[max(0,best_r-margin):best_r+margin,
                   max(0,best_c-margin):best_c+margin]
        weights = reg ** 2
        total = weights.sum()
        ry = np.arange(max(0,best_r-margin), max(0,best_r-margin)+reg.shape[0])
        rx = np.arange(max(0,best_c-margin), max(0,best_c-margin)+reg.shape[1])
        refined_y = int((weights.sum(axis=1) * ry).sum() / total)
        refined_x = int((weights.sum(axis=0) * rx).sum() / total)

        # Angular position relative to Sun centre
        # Positive x = west (right), positive y = south (down) in COR2/C3 frame
        sun_cy, sun_cx = SUN_CENTRE['c3']
        dx_px = refined_x - sun_cx
        dy_px = refined_y - sun_cy
        ps = PLATE_SCALE['c3']
        dx_deg = dx_px * ps / 3600.0
        dy_deg = dy_px * ps / 3600.0
        angular_sep_deg = ((dx_deg**2 + dy_deg**2) ** 0.5)

        # Anti-solar angle — direction from Sun to comet in image plane
        import math
        anti_solar_angle = math.degrees(math.atan2(-dy_px, -dx_px)) % 360

        log.info(f"C3 refined centroid: ({refined_y},{refined_x}) "
                 f"sep={angular_sep_deg:.2f}deg anti_solar={anti_solar_angle:.1f}deg")

        return {
            'instrument':        'c3',
            'centroid_y':        refined_y,
            'centroid_x':        refined_x,
            'dx_px':             dx_px,
            'dy_px':             dy_px,
            'angular_sep_deg':   round(angular_sep_deg, 4),
            'anti_solar_angle':  round(anti_solar_angle, 2),
            'peak_brightness':   round(peak_val, 3),
            'compactness_ratio': round(compactness, 3),
            'plate_scale_arcsec_px': PLATE_SCALE['c3'],
        }

    except Exception as e:
        log.error(f"detect_c3_centroid: {e}")
        return None


# ── VELOCITY COMPUTATION ─────────────────────────────────────────────────
def compute_velocity(current_detection, instrument):
    """
    Loads previous centroid from trajectory log and computes:
    - Pixel displacement between frames
    - Angular velocity in deg/hr
    - Position angle of motion
    Returns velocity dict or None if no previous position available.
    """
    try:
        import math
        if not TRAJ_PATH.exists():
            return None

        # Find last entry for this instrument
        lines = TRAJ_PATH.read_text().strip().splitlines()
        prev = None
        for line in reversed(lines):
            try:
                entry = __import__('json').loads(line)
                if entry.get('instrument') == instrument:
                    prev = entry
                    break
            except Exception:
                continue

        if prev is None:
            return None

        # Compute displacement
        dy = current_detection['centroid_y'] - prev['centroid_y']
        dx = current_detection['centroid_x'] - prev['centroid_x']
        displacement_px = (dx**2 + dy**2) ** 0.5

        # Time delta
        from datetime import datetime, timezone
        t_cur = datetime.now(timezone.utc)
        try:
            t_prev = datetime.fromisoformat(prev['utc'].replace('Z','+00:00'))
            dt_hr = (t_cur - t_prev).total_seconds() / 3600.0
        except Exception:
            dt_hr = 0.5  # assume 30 min if parse fails

        if dt_hr < 0.05:
            return None  # too soon, same frame

        # Angular velocity
        ps = PLATE_SCALE.get(instrument, 58.8)
        displacement_deg = displacement_px * ps / 3600.0
        angular_velocity_deg_hr = displacement_deg / dt_hr if dt_hr > 0 else 0

        # Position angle of motion (degrees, N=0, E=90)
        motion_angle = math.degrees(math.atan2(-dy, dx)) % 360

        log.info(f"{instrument} velocity: {displacement_px:.1f}px "
                 f"= {displacement_deg:.4f}deg in {dt_hr:.2f}hr "
                 f"= {angular_velocity_deg_hr:.4f}deg/hr "
                 f"PA={motion_angle:.1f}deg")

        return {
            'displacement_px':         round(displacement_px, 3),
            'displacement_deg':        round(displacement_deg, 6),
            'dt_hr':                   round(dt_hr, 4),
            'angular_velocity_deg_hr': round(angular_velocity_deg_hr, 6),
            'motion_position_angle':   round(motion_angle, 2),
            'prev_centroid_y':         prev['centroid_y'],
            'prev_centroid_x':         prev['centroid_x'],
        }

    except Exception as e:
        log.error(f"compute_velocity: {e}")
        return None


# ── TRAJECTORY LOG ────────────────────────────────────────────────────────
def write_trajectory_entry(detection, velocity, cag_uuid, utc_str):
    """
    Appends one entry to hale_trajectory.jsonl.
    Each entry = one centroid position + velocity + UUID.
    This is the raw material for trajectory line fitting.
    Keep last 500 entries (rolling).
    """
    import json
    entry = {
        'utc':              utc_str,
        'instrument':       detection.get('instrument', 'unknown'),
        'centroid_y':       detection['centroid_y'],
        'centroid_x':       detection['centroid_x'],
        'angular_sep_deg':  detection.get('angular_sep_deg'),
        'anti_solar_angle': detection.get('anti_solar_angle'),
        'peak_brightness':  detection.get('peak_brightness'),
        'compactness':      detection.get('compactness_ratio'),
        'velocity':         velocity,
        'cag_uuid':         cag_uuid,
    }
    with open(TRAJ_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    # Rolling window
    try:
        lines = TRAJ_PATH.read_text().splitlines()
        if len(lines) > 500:
            TRAJ_PATH.write_text('\n'.join(lines[-500:]) + '\n')
    except Exception:
        pass
    log.info(f"Trajectory entry: {detection['instrument']} "
             f"({detection['centroid_y']},{detection['centroid_x']}) "
             f"uuid={cag_uuid}")


# ── ANNOTATE C3 FRAME ────────────────────────────────────────────────────
def annotate_c3_frame(image_path, output_path, detection, trajectory_history=None):
    """
    Draws circle on C3 comet detection.
    Optionally draws trajectory trail from previous positions.
    """
    try:
        from PIL import Image, ImageDraw
        import json
        img = Image.open(image_path).copy()
        draw = ImageDraw.Draw(img)
        cy = detection['centroid_y']
        cx = detection['centroid_x']
        r = 14  # fixed radius for C3 — source is compact at entry

        # Draw trajectory trail first (underneath circles)
        if trajectory_history and len(trajectory_history) > 1:
            pts = [(e['centroid_x'], e['centroid_y'])
                   for e in trajectory_history[-12:]  # last 12 points = 6 hours
                   if e.get('instrument') == 'c3']
            if len(pts) > 1:
                for i in range(len(pts)-1):
                    # Fade older points
                    alpha = int(80 + 120 * (i / len(pts)))
                    draw.line([pts[i], pts[i+1]],
                              fill=(0, alpha, int(alpha*0.7)), width=1)
                # Mark each historical point
                for pt in pts[:-1]:
                    draw.ellipse([pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2],
                                 fill=(0, 160, 120))

        # Current position — double circle
        draw.ellipse([cx-r-4, cy-r-4, cx+r+4, cy+r+4],
                     outline=(0, 200, 255), width=1)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                     outline=(0, 160, 200), width=1)
        # Crosshair
        gap, arm = 3, 6
        draw.line([(cx-arm-gap, cy), (cx-gap, cy)], fill=(0,200,255), width=1)
        draw.line([(cx+gap, cy), (cx+arm+gap, cy)], fill=(0,200,255), width=1)
        draw.line([(cx, cy-arm-gap), (cx, cy-gap)], fill=(0,200,255), width=1)
        draw.line([(cx, cy+gap), (cx, cy+arm+gap)], fill=(0,200,255), width=1)

        # Anti-solar direction line from Sun centre
        import math
        sun_cy, sun_cx = SUN_CENTRE['c3']
        angle_rad = math.radians(detection.get('anti_solar_angle', 0))
        line_len = 30
        ex = int(cx + line_len * math.cos(angle_rad))
        ey = int(cy - line_len * math.sin(angle_rad))
        draw.line([(cx, cy), (ex, ey)], fill=(255, 200, 0), width=1)

        img.save(output_path, quality=92)
        os.chmod(output_path, 0o644)
        try:
            import pwd, grp
            os.chown(output_path,
                     pwd.getpwnam('www-data').pw_uid,
                     grp.getgrnam('www-data').gr_gid)
        except Exception:
            pass
        log.info(f"C3 annotated frame saved: {output_path}")
        return True
    except Exception as e:
        log.error(f"annotate_c3_frame: {e}")
        return False

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
        if any(name.endswith(s) for s in ['_current', '_previous', '_annotated', '_traj']):
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

    # 3b. C3 centroid detection + velocity + trajectory
    c3_detection = None
    c3_velocity  = None
    c3_path = FRAME_DIR / 'c3_current.jpg'
    if c3_path.exists():
        c3_detection = detect_c3_centroid(str(c3_path))
        if c3_detection:
            c3_velocity = compute_velocity(c3_detection, 'c3')
            # Wolfram UUID for C3 detection
            wl_c3 = f"""
Module[{{uuid}},
  uuid = ToString[CreateUUID[]];
  ExportString[{{
    "instrument"      -> "c3",
    "centroid_x"      -> {c3_detection['centroid_x']},
    "centroid_y"      -> {c3_detection['centroid_y']},
    "angular_sep_deg" -> {c3_detection['angular_sep_deg']},
    "anti_solar_ang"  -> {c3_detection['anti_solar_angle']},
    "peak_brightness" -> {c3_detection['peak_brightness']},
    "compactness"     -> {c3_detection['compactness_ratio']},
    "method"          -> "weighted_centroid_v1",
    "cag_uuid"        -> uuid
  }}, "JSON"]
]
"""
            try:
                wl_result = subprocess.run(
                    ['wolframscript', '-code', wl_c3],
                    capture_output=True, text=True, timeout=90
                )
                if wl_result.returncode == 0:
                    import json as _json
                    parsed = _json.loads(wl_result.stdout.strip())
                    if isinstance(parsed, list):
                        d = {item[0]: item[1] for item in parsed
                             if isinstance(item, list)}
                        c3_uuid = d.get('cag_uuid')
                    else:
                        c3_uuid = parsed.get('cag_uuid')
                    c3_detection['cag_uuid'] = c3_uuid
                    log.info(f"C3 detection uuid={c3_uuid}")
                else:
                    c3_detection['cag_uuid'] = None
            except Exception as e:
                log.error(f"C3 Wolfram UUID: {e}")
                c3_detection['cag_uuid'] = None

            # Write trajectory entry
            write_trajectory_entry(c3_detection, c3_velocity,
                                   c3_detection.get('cag_uuid'), utc_now)

            # Load trajectory history for annotation
            traj_history = []
            if TRAJ_PATH.exists():
                for line in TRAJ_PATH.read_text().strip().split('\n'):
                    try:
                        traj_history.append(__import__('json').loads(line))
                    except Exception:
                        pass

            # Annotate C3 frame
            c3_ann = FRAME_DIR / 'c3_annotated.jpg'
            annotate_c3_frame(str(c3_path), str(c3_ann),
                              c3_detection, traj_history)

    # Also write COR2 centroid to trajectory if detected
    if centroid_result and centroid_result.get('cag_uuid'):
        cor2_velocity = compute_velocity(centroid_result, 'cor2')
        centroid_result['velocity'] = cor2_velocity
        write_trajectory_entry(centroid_result, cor2_velocity,
                               centroid_result.get('cag_uuid'), utc_now)

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
    entry['comet_detection_cor2'] = centroid_result if centroid_result else {
        'centroid_x': None, 'centroid_y': None, 'cag_uuid': None,
        'note': 'no_detection'
    }
    entry['comet_detection_c3'] = c3_detection if c3_detection else {
        'centroid_x': None, 'centroid_y': None, 'cag_uuid': None,
        'note': 'no_detection'
    }
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
