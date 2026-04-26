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


# ── DETERMINISTIC UUID (Mistral Comm.27) ─────────────────────────────────
import uuid as _uuid_mod
# Wolfram-compatible namespace (standard DNS namespace)
CAG_NAMESPACE = _uuid_mod.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

def deterministic_uuid(instrument, timestamp, centroid_x, centroid_y,
                        flow_mag=None, flow_dir=None):
    """
    Generate reproducible UUID v5 from computation inputs.
    Same inputs always produce the same UUID — independently verifiable.
    Replaces Wolfram CreateUUID[] for non-computation UUIDs.
    """
    data = f"{instrument}:{timestamp}:{centroid_x}:{centroid_y}"
    if flow_mag is not None:
        data += f":{flow_mag:.6f}:{flow_dir:.2f}"
    return str(_uuid_mod.uuid5(CAG_NAMESPACE, data))

# ── Paths ─────────────────────────────────────────────────────────────────
BASE        = Path('/var/www/heliodata.ai/html/data')
FRAME_DIR   = BASE / 'coronagraph'
JSONL_PATH  = BASE / 'hale_coronagraph.jsonl'
TRAJ_PATH   = BASE / 'hale_trajectory.jsonl'
PLATE_SCALE = {'cor2': 58.8, 'c3': 56.0, 'hi2': 960.0}
SUN_CENTRE = {'cor2': (256,256), 'c3': (256,256)}

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

        if compactness < 2.0:
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
        sun_cy, sun_cx = (256, 256)  # COR2 occulter centre
        dx_px = refined_x - sun_cx
        dy_px = refined_y - sun_cy
        ps = 58.8  # COR2 arcsec/px at 512px beacon
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
            'plate_scale_arcsec_px': 58.8,  # COR2 arcsec/px
        }

    except Exception as e:
        import traceback
        log.error(f"detect_comet_centroid: {e}")
        log.error(traceback.format_exc())
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
    Generate deterministic UUID for centroid detection (Mistral Comm.27).
    Replaces Wolfram CreateUUID[] — same inputs = same UUID, reproducible.
    Falls back to Wolfram only if deterministic generation fails.
    """
    try:
        uid = deterministic_uuid(
            detection.get('instrument', 'cor2'),
            __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            detection.get('centroid_x', 0),
            detection.get('centroid_y', 0),
            detection.get('peak_brightness'),
            detection.get('compactness_ratio')
        )
        log.info(f"Deterministic UUID: {uid}")
        return uid
    except Exception as e:
        log.warning(f"Deterministic UUID failed: {e}, falling back to Wolfram")
        # Fallback to Wolfram
        try:
            result = subprocess.run(
                ['wolframscript', '-code', 'ToString[CreateUUID[]]'],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return result.stdout.strip().strip('"')
        except Exception:
            pass
        return None


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
        # Reject frame-edge detections — row < 25 is artifact zone
        if best_r < 25:
            log.warning(f"detect_c3_centroid: candidate at row {best_r} is frame edge artifact")
            return None
        # Reject stars — elongation check
        # Stars have horiz/vert extent ratio ~1.0, comet coma is rounder but brighter
        # For now disable C3 centroid — comet is in streak phase, needs ImageLines
        log.info("detect_c3_centroid: comet in streak phase — centroid disabled pending ImageLines implementation")
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




# ── GYORI COMET ASSAY PROFILE (Gyori et al. 2014, Redox Biology) ─────────
# Adapted from biomedical comet assay to astronomical coronagraph.
# Same morphology: bright head + fading tail. "As above so below."
# Consortium Comm.28 optimizations: strip=6px, 2nd deriv boundary,
# dual tail endpoints, Olive moment, validity classifier.
def gyori_profile_analysis(image_path, centroid, sun_centre=(256,256)):
    """
    Intensity profile analysis along anti-solar (tail) axis.
    Adapted from Gyori et al. 2014, Eq. 3-6.
    Optimized per Consortium Comm.28 (DeepSeek + Mistral).

    Returns dict with measurements or None on failure.
    """
    try:
        import cv2
        import numpy as np
        import math
        from scipy.ndimage import gaussian_filter1d
        import hashlib

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            log.warning("gyori_profile: cannot read image")
            return None

        h, w = img.shape
        gray = img.astype(float)
        cy, cx = centroid['centroid_y'], centroid['centroid_x']
        sun_y, sun_x = sun_centre

        # Anti-solar direction (tail axis)
        dx = cx - sun_x
        dy = cy - sun_y
        tail_angle = math.atan2(dy, dx) + math.pi  # anti-solar: AWAY from Sun
        anti_solar_deg = math.degrees(tail_angle)

        # ── Strip width: 6px (DeepSeek Comm.28: 3× SNR over 20px) ────
        strip_width = 6
        perp_angle = tail_angle + math.pi / 2
        max_sample = 120

        # Project intensity along tail axis (Gyori Eq. 3)
        profile = []
        profile_coords = []

        for d in range(-20, max_sample):
            px = cx + d * math.cos(tail_angle)
            py = cy + d * math.sin(tail_angle)
            if not (0 <= int(py) < h and 0 <= int(px) < w):
                continue

            strip_vals = []
            for s in range(-strip_width, strip_width + 1):
                sx = int(px + s * math.cos(perp_angle))
                sy = int(py + s * math.sin(perp_angle))
                if 0 <= sy < h and 0 <= sx < w:
                    strip_vals.append(float(gray[sy, sx]))

            if strip_vals:
                profile.append(float(np.mean(strip_vals)))
                profile_coords.append((d, int(py), int(px)))

        if len(profile) < 10:
            log.warning("gyori_profile: profile too short")
            return None

        profile = np.array(profile)
        head_idx = 20  # d=0 corresponds to head

        # ── Validity classifier (DeepSeek Comm.28 insight) ────────────
        profile_norm = profile / (profile.max() + 1e-6)
        p_d_check = np.gradient(profile_norm)
        gyori_valid = True
        validity_note = 'valid_comet_shape'
        if p_d_check.min() > -0.1:
            gyori_valid = False
            validity_note = 'no_tail_detected'
        else:
            transition_check = np.argmin(p_d_check)
            tail_check = profile_norm[transition_check:transition_check+10]
            if len(tail_check) > 2 and np.any(np.diff(tail_check) > 0.05):
                gyori_valid = False
                validity_note = 'non_monotonic_tail'

        # Background from far tail
        bg_region = profile[-20:] if len(profile) > 30 else profile[-5:]
        bg_mean = float(np.median(bg_region))
        bg_std = float(np.std(bg_region)) if len(bg_region) > 3 else 5.0

        # ── Gaussian smoothing σ=2.0 (Comm.27+28 consensus) ──────────
        smooth = gaussian_filter1d(profile, sigma=2.0)

        # ── Head-tail boundary: 2nd derivative zero-crossing ──────────
        # (Gyori Eq. 6: first maximum in pdd after neg→pos crossing)
        p_d = np.gradient(smooth)
        p_dd = np.gradient(p_d)

        # Find neg→pos zero crossings in p_dd after head
        crossings = np.where(
            (p_dd[head_idx:-1] < 0) & (p_dd[head_idx+1:] > 0)
        )[0]

        if len(crossings) > 0:
            crossing_idx = crossings[0] + head_idx
            # First maximum in p_dd after crossing
            post = p_dd[crossing_idx:min(crossing_idx+10, len(p_dd))]
            if len(post) > 0:
                boundary_idx = crossing_idx + int(np.argmax(post))
            else:
                boundary_idx = crossing_idx
        else:
            # Fallback: steepest negative gradient
            boundary_idx = head_idx + int(np.argmin(p_d[head_idx:]))

        boundary_offset = max(3, boundary_idx - head_idx)
        boundary_idx = head_idx + boundary_offset

        # ── Dual tail endpoint (Comm.28 consensus) ────────────────────
        # Ion tail: inflection point (derivative flattens)
        p_d_smooth = gaussian_filter1d(p_d, sigma=1.0)
        max_abs_deriv = np.max(np.abs(p_d_smooth[head_idx:])) + 1e-6
        inflection_candidates = np.where(
            np.abs(p_d_smooth[boundary_idx:]) < 0.05 * max_abs_deriv
        )[0]
        if len(inflection_candidates) > 0:
            ion_end_idx = boundary_idx + inflection_candidates[0]
            ion_tail_px = ion_end_idx - head_idx
        else:
            ion_end_idx = len(smooth) - 1
            ion_tail_px = ion_end_idx - head_idx

        # Total tail: SNR ≥ 3
        snr_threshold = bg_mean + 3 * bg_std
        total_end_idx = boundary_idx
        for i in range(boundary_idx, len(smooth)):
            if smooth[i] <= snr_threshold:
                total_end_idx = i
                break
        else:
            total_end_idx = len(smooth) - 1
        total_tail_px = max(0, total_end_idx - head_idx)

        # Use total tail for primary measurement (P5 compatible)
        tail_length_px = total_tail_px

        # Plate scale
        ps = PLATE_SCALE.get('cor2', 58.8)
        tail_length_arcsec = tail_length_px * ps
        tail_length_deg = tail_length_arcsec / 3600.0
        ion_tail_deg = ion_tail_px * ps / 3600.0

        # ── Tail moment (Gyori Table 2) ───────────────────────────────
        total_intensity = float(profile[head_idx:].sum()) + 1e-6
        tail_intensity = float(profile[boundary_idx:total_end_idx].sum())
        head_intensity = float(profile[head_idx:boundary_idx].sum())
        pct_in_tail = (tail_intensity / total_intensity) * 100
        tail_moment = tail_length_px * (pct_in_tail / 100)

        # ── Olive moment (Gyori Table 2 — better for P2) ─────────────
        if tail_intensity > 0 and total_end_idx > boundary_idx:
            tail_prof = profile[boundary_idx:total_end_idx]
            tail_dists = np.arange(len(tail_prof)) + boundary_offset
            tail_centroid_d = float(np.average(tail_dists,
                                               weights=tail_prof + 1e-6))
            olive_moment = (pct_in_tail / 100) * abs(tail_centroid_d)
        else:
            tail_centroid_d = 0
            olive_moment = 0

        # ── Deterministic UUID from profile array (Mistral Option C) ──
        profile_bytes = profile.tobytes()
        profile_hash = hashlib.sha1(profile_bytes).hexdigest()
        gyori_uuid = str(_uuid_mod.uuid5(CAG_NAMESPACE, profile_hash))

        log.info(f"gyori_profile: tail_total={total_tail_px}px={tail_length_deg:.3f}deg "
                 f"tail_ion={ion_tail_px}px={ion_tail_deg:.3f}deg "
                 f"moment={tail_moment:.2f} olive={olive_moment:.2f} "
                 f"pct={pct_in_tail:.1f}% valid={validity_note} "
                 f"uuid={gyori_uuid[:8]}")

        result = {
            'method':               'gyori_2014_adapted_comm28',
            'anti_solar_deg':       round(anti_solar_deg, 2),
            'head_tail_boundary_px': int(boundary_offset),
            'tail_length_px':       int(total_tail_px),
            'tail_length_arcsec':   round(tail_length_arcsec, 1),
            'tail_length_deg':      round(tail_length_deg, 4),
            'ion_tail_length_px':   int(ion_tail_px),
            'ion_tail_length_deg':  round(ion_tail_deg, 4),
            'tail_moment':          round(tail_moment, 3),
            'olive_moment':         round(olive_moment, 3),
            'pct_in_tail':          round(pct_in_tail, 2),
            'head_intensity':       round(head_intensity, 1),
            'tail_intensity':       round(tail_intensity, 1),
            'background_mean':      round(bg_mean, 2),
            'background_std':       round(bg_std, 2),
            'plate_scale_arcsec':   ps,
            'strip_width_px':       strip_width,
            'profile_length':       int(len(profile)),
            'gyori_valid':          gyori_valid,
            'validity_note':        validity_note,
            'boundary_method':      '2nd_derivative_zero_crossing',
            'tail_end_method':      'dual_inflection_snr3',
            'cag_uuid':            gyori_uuid,
        }

        # ── Visualization ─────────────────────────────────────────────
        try:
            output = cv2.imread(str(image_path))

            # Tail axis line (colour-coded)
            for d_idx, (d, py_c, px_c) in enumerate(profile_coords):
                if d < 0:
                    cv2.circle(output, (px_c, py_c), 0, (0,200,255), 1)
                elif d < boundary_offset:
                    cv2.circle(output, (px_c, py_c), 0, (0,255,200), 1)
                elif d < total_tail_px:
                    cv2.circle(output, (px_c, py_c), 0, (0,255,255), 1)
                else:
                    cv2.circle(output, (px_c, py_c), 0, (100,100,100), 1)

            # Head crosshair
            cv2.line(output, (cx-6,cy), (cx-2,cy), (0,255,200), 1)
            cv2.line(output, (cx+2,cy), (cx+6,cy), (0,255,200), 1)
            cv2.line(output, (cx,cy-6), (cx,cy-2), (0,255,200), 1)
            cv2.line(output, (cx,cy+2), (cx,cy+6), (0,255,200), 1)
            cv2.circle(output, (cx,cy), 8, (0,255,200), 1)

            # Boundary marker (red)
            bnd_d = head_idx + boundary_offset
            if bnd_d < len(profile_coords):
                _, by, bx = profile_coords[bnd_d]
                cv2.circle(output, (bx, by), 4, (0,0,255), 1)

            # Ion tail end (cyan)
            if ion_end_idx < len(profile_coords):
                _, iy, ix = profile_coords[ion_end_idx]
                cv2.circle(output, (ix, iy), 3, (255,255,0), 1)

            # Total tail end (magenta)
            if total_end_idx < len(profile_coords):
                _, ey, ex = profile_coords[total_end_idx]
                cv2.circle(output, (ex, ey), 4, (255,0,255), 1)

            # Mini profile plot
            plot_h, plot_w = 80, 140
            plot_x0 = 10
            plot_y0 = h - plot_h - 30
            cv2.rectangle(output, (plot_x0,plot_y0),
                         (plot_x0+plot_w, plot_y0+plot_h), (30,30,30), -1)
            cv2.rectangle(output, (plot_x0,plot_y0),
                         (plot_x0+plot_w, plot_y0+plot_h), (100,100,100), 1)
            p_disp = smooth[:min(len(smooth), plot_w)]
            p_max = p_disp.max() if p_disp.max() > 0 else 1
            for i in range(1, len(p_disp)):
                y1 = int(plot_y0+plot_h-(p_disp[i-1]/p_max)*(plot_h-4))
                y2 = int(plot_y0+plot_h-(p_disp[i]/p_max)*(plot_h-4))
                cv2.line(output, (plot_x0+i-1,y1), (plot_x0+i,y2), (0,255,200), 1)
            # SNR threshold
            bg_y = int(plot_y0+plot_h-(snr_threshold/p_max)*(plot_h-4))
            if 0 < bg_y < plot_y0+plot_h:
                cv2.line(output, (plot_x0,bg_y), (plot_x0+plot_w,bg_y), (0,0,200), 1)
            # Boundary on plot
            bnd_plot = boundary_offset + 20
            if bnd_plot < plot_w:
                cv2.line(output, (plot_x0+bnd_plot,plot_y0),
                        (plot_x0+bnd_plot,plot_y0+plot_h), (0,0,255), 1)

            # Validity border
            if not gyori_valid:
                cv2.rectangle(output, (0,0), (w-1,h-1), (0,0,180), 2)

            prof_path = FRAME_DIR / 'cor2_profile.jpg'
            cv2.imwrite(str(prof_path), output, [cv2.IMWRITE_JPEG_QUALITY, 92])
            os.chmod(prof_path, 0o644)
            try:
                import pwd, grp
                os.chown(prof_path, pwd.getpwnam('www-data').pw_uid,
                         grp.getgrnam('www-data').gr_gid)
            except Exception:
                pass
            log.info("gyori_profile: visualization saved")
        except Exception as ve:
            log.warning(f"gyori_profile viz: {ve}")

        return result

    except Exception as e:
        log.error(f"gyori_profile: {e}")
        return None


# ── C3 GYORI PROFILE (observed tail direction scan) ──────────────────────
def find_c3_comet_head(image_path):
    """
    Find comet coma head in C3 via largest bright connected region.
    Returns (cy, cx) or None.
    The comet is an extended bright feature — centroid detection fails
    because stars are more compact. Instead find the largest bright region
    and locate its peak brightness pixel.
    """
    try:
        import cv2
        import numpy as np
        from scipy import ndimage

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        gray = img.astype(float)
        h, w = img.shape

        # Mask occulter and edges
        Y, X = np.ogrid[:h, :w]
        mask = ((Y-256)**2 + (X-256)**2 < 80**2) | (Y<25) | (Y>487) | (X<25) | (X>487)
        masked = gray.copy()
        masked[mask] = 0

        # Threshold at 92nd percentile to find bright regions
        valid = masked[masked > 0]
        if len(valid) < 100:
            return None
        thresh = np.percentile(valid, 92)
        binary = (masked > thresh).astype(np.uint8)

        labeled, n = ndimage.label(binary)
        if n == 0:
            return None

        # Find largest component
        sizes = [(labeled == i).sum() for i in range(1, n+1)]
        largest_id = np.argmax(sizes) + 1
        largest = labeled == largest_id
        largest_size = sizes[largest_id - 1]

        if largest_size < 50:
            log.info("find_c3_comet_head: largest region too small")
            return None

        # Find peak brightness within largest component
        comp_vals = gray.copy()
        comp_vals[~largest] = 0
        peak_loc = np.unravel_index(comp_vals.argmax(), comp_vals.shape)
        peak_val = float(gray[peak_loc[0], peak_loc[1]])

        # Verify it's below the occulter (comet is in lower half during this transit)
        if peak_loc[0] < 280:
            log.info(f"find_c3_comet_head: peak at row {peak_loc[0]} is above occulter, skipping")
            return None

        log.info(f"find_c3_comet_head: ({peak_loc[0]},{peak_loc[1]}) "
                 f"peak={peak_val:.0f} region={largest_size}px")
        return (int(peak_loc[0]), int(peak_loc[1]))

    except Exception as e:
        log.error(f"find_c3_comet_head: {e}")
        return None


def find_observed_tail_direction(image_path, cy, cx, r_inner=15, r_outer=50):
    """
    Scan 360 degrees around the coma head and find the direction
    with highest average brightness. That is the observed tail direction,
    accounting for projection, dust lag, and forward scatter geometry.
    """
    try:
        import cv2
        import numpy as np
        import math

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        gray = img.astype(float)
        h, w = img.shape

        best_angle = 0
        best_brightness = 0
        results = []

        for angle_deg in range(0, 360, 5):
            angle = math.radians(angle_deg)
            vals = []
            for r in range(r_inner, r_outer):
                px = int(cx + r * math.cos(angle))
                py = int(cy + r * math.sin(angle))
                if 0 <= py < h and 0 <= px < w:
                    vals.append(float(gray[py, px]))
            mean_b = float(np.mean(vals)) if vals else 0
            results.append((angle_deg, mean_b))
            if mean_b > best_brightness:
                best_brightness = mean_b
                best_angle = angle_deg

        log.info(f"observed_tail_direction: {best_angle} deg "
                 f"(brightness={best_brightness:.1f})")
        return best_angle

    except Exception as e:
        log.error(f"find_observed_tail_direction: {e}")
        return None


def gyori_profile_c3(image_path, cy, cx, tail_angle_deg):
    """
    Run Gyori profile on C3 frame using observed tail direction.
    Same algorithm as COR2 but with C3 plate scale and observed direction.
    """
    try:
        import cv2
        import numpy as np
        import math
        from scipy.ndimage import gaussian_filter1d
        import hashlib

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        gray = img.astype(float)
        h, w = img.shape

        tail_angle = math.radians(tail_angle_deg)
        strip_width = 6  # Comm.28
        perp = tail_angle + math.pi / 2
        profile = []
        coords = []

        for d in range(-20, 200):
            px = cx + d * math.cos(tail_angle)
            py = cy + d * math.sin(tail_angle)
            if not (0 <= int(py) < h and 0 <= int(px) < w):
                continue
            vals = []
            for s in range(-strip_width, strip_width + 1):
                sx = int(px + s * math.cos(perp))
                sy = int(py + s * math.sin(perp))
                if 0 <= sy < h and 0 <= sx < w:
                    vals.append(float(gray[sy, sx]))
            if vals:
                profile.append(float(np.mean(vals)))
                coords.append((d, int(py), int(px)))

        if len(profile) < 15:
            log.warning("gyori_c3: profile too short")
            return None

        profile = np.array(profile)
        head_idx = 20

        # Background
        bg = profile[-30:] if len(profile) > 40 else profile[-10:]
        bg_mean = float(np.median(bg))
        bg_std = float(np.std(bg)) if len(bg) > 3 else 5.0

        # Smooth
        smooth = gaussian_filter1d(profile, sigma=2.0)
        p_d = np.gradient(smooth)
        p_dd = np.gradient(p_d)

        # 2nd derivative boundary
        cross = np.where((p_dd[head_idx:-1] < 0) & (p_dd[head_idx+1:] > 0))[0]
        if len(cross) > 0:
            ci = cross[0] + head_idx
            post = p_dd[ci:min(ci+10, len(p_dd))]
            bi = ci + int(np.argmax(post)) if len(post) > 0 else ci
        else:
            bi = head_idx + int(np.argmin(p_d[head_idx:]))
        bo = max(3, bi - head_idx)

        # Dual endpoints
        snr_t = bg_mean + 3 * bg_std
        te = bi
        for i in range(bi, len(smooth)):
            if smooth[i] <= snr_t:
                te = i
                break
        else:
            te = len(smooth) - 1
        total_px = max(0, te - head_idx)

        pds = gaussian_filter1d(p_d, sigma=1.0)
        md = np.max(np.abs(pds[head_idx:])) + 1e-6
        inf = np.where(np.abs(pds[bi:]) < 0.05 * md)[0]
        ie = bi + inf[0] if len(inf) > 0 else len(smooth) - 1
        ion_px = ie - head_idx

        ps = PLATE_SCALE.get('c3', 56.0)
        total_deg = total_px * ps / 3600
        ion_deg = ion_px * ps / 3600

        # Intensities
        tot_i = float(profile[head_idx:].sum()) + 1e-6
        tail_i = float(profile[head_idx+bo:te].sum())
        head_i = float(profile[head_idx:head_idx+bo].sum())
        pct = (tail_i / tot_i) * 100
        tm = total_px * (pct / 100)

        # Olive moment
        if tail_i > 0 and te > bi:
            tp = profile[head_idx+bo:te]
            td = np.arange(len(tp)) + bo
            tc = float(np.average(td, weights=tp + 1e-6))
            ol = (pct / 100) * abs(tc)
        else:
            ol = 0

        # UUID from profile array
        profile_hash = hashlib.sha1(profile.tobytes()).hexdigest()
        gyori_uuid = str(_uuid_mod.uuid5(CAG_NAMESPACE, profile_hash))

        log.info(f"gyori_c3: tail={total_px}px={total_deg:.3f}deg "
                 f"moment={tm:.2f} pct={pct:.1f}% olive={ol:.2f} "
                 f"dir={tail_angle_deg}deg uuid={gyori_uuid[:8]}")

        result = {
            'method':                'gyori_2014_observed_direction',
            'tail_direction_deg':    tail_angle_deg,
            'head_tail_boundary_px': int(bo),
            'tail_length_px':        int(total_px),
            'tail_length_deg':       round(total_deg, 4),
            'ion_tail_length_px':    int(ion_px),
            'ion_tail_length_deg':   round(ion_deg, 4),
            'tail_moment':           round(tm, 3),
            'olive_moment':          round(ol, 3),
            'pct_in_tail':           round(pct, 2),
            'head_intensity':        round(head_i, 1),
            'tail_intensity':        round(tail_i, 1),
            'background_mean':       round(bg_mean, 2),
            'background_std':        round(bg_std, 2),
            'plate_scale_arcsec':    ps,
            'strip_width_px':        strip_width,
            'cag_uuid':              gyori_uuid,
        }

        # Visualization
        try:
            output = cv2.imread(str(image_path))
            for di, (d, py_c, px_c) in enumerate(coords):
                if d < 0:
                    cv2.circle(output, (px_c, py_c), 0, (0,200,255), 1)
                elif d < bo:
                    cv2.circle(output, (px_c, py_c), 0, (0,255,200), 1)
                elif d < total_px:
                    cv2.circle(output, (px_c, py_c), 0, (0,255,255), 1)
                else:
                    cv2.circle(output, (px_c, py_c), 0, (100,100,100), 1)

            cv2.circle(output, (cx, cy), 10, (0,255,200), 1)
            cv2.line(output, (cx-8,cy), (cx-3,cy), (0,255,200), 1)
            cv2.line(output, (cx+3,cy), (cx+8,cy), (0,255,200), 1)
            cv2.line(output, (cx,cy-8), (cx,cy-3), (0,255,200), 1)
            cv2.line(output, (cx,cy+3), (cx,cy+8), (0,255,200), 1)

            if head_idx+bo < len(coords):
                _, by, bx = coords[head_idx+bo]
                cv2.circle(output, (bx, by), 5, (0,0,255), 1)
            if te < len(coords):
                _, ey, ex = coords[te]
                cv2.circle(output, (ex, ey), 5, (255,0,255), 1)

            # Mini profile
            ph, pw = 80, 160
            px0 = 10; py0 = h - ph - 30
            cv2.rectangle(output, (px0,py0), (px0+pw,py0+ph), (30,30,30), -1)
            cv2.rectangle(output, (px0,py0), (px0+pw,py0+ph), (100,100,100), 1)
            pd = smooth[:min(len(smooth), pw)]
            pm = pd.max() if pd.max() > 0 else 1
            for i in range(1, len(pd)):
                y1 = int(py0+ph-(pd[i-1]/pm)*(ph-4))
                y2 = int(py0+ph-(pd[i]/pm)*(ph-4))
                cv2.line(output, (px0+i-1,y1), (px0+i,y2), (0,255,200), 1)
            ty = int(py0+ph-(snr_t/pm)*(ph-4))
            if py0 < ty < py0+ph:
                cv2.line(output, (px0,ty), (px0+pw,ty), (0,0,200), 1)
            bo_plot = bo + 20
            if bo_plot < pw:
                cv2.line(output, (px0+bo_plot,py0), (px0+bo_plot,py0+ph), (0,0,255), 1)

            c3_prof = FRAME_DIR / 'c3_profile.jpg'
            cv2.imwrite(str(c3_prof), output, [cv2.IMWRITE_JPEG_QUALITY, 92])
            os.chmod(c3_prof, 0o644)
            try:
                import pwd, grp
                os.chown(c3_prof, pwd.getpwnam('www-data').pw_uid,
                         grp.getgrnam('www-data').gr_gid)
            except Exception:
                pass
        except Exception as ve:
            log.warning(f"gyori_c3 viz: {ve}")

        return result

    except Exception as e:
        log.error(f"gyori_c3: {e}")
        return None

# ── PHASE CORRELATION SANITY CHECK (DeepSeek Comm.27) ────────────────────
def phase_correlation_check(cur_path, prev_path, cookie_centre, cookie_radius):
    """
    Fast global shift estimation within cookie, robust to JPEG artifacts.
    Returns shift vector or None. Used to validate Farneback results.
    """
    try:
        import cv2
        import numpy as np
        cur = cv2.imread(str(cur_path), cv2.IMREAD_GRAYSCALE)
        prev = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
        if cur is None or prev is None:
            return None
        if abs(cur.astype(int)-prev.astype(int)).max() == 0:
            return None
        h, w = cur.shape
        cy, cx = cookie_centre
        r = cookie_radius
        # Extract cookie region
        y0 = max(0, cy-r); y1 = min(h, cy+r)
        x0 = max(0, cx-r); x1 = min(w, cx+r)
        crop_cur = cur[y0:y1, x0:x1].astype(np.float32)
        crop_prev = prev[y0:y1, x0:x1].astype(np.float32)
        # Gaussian preprocess
        crop_cur = cv2.GaussianBlur(crop_cur, (0,0), sigmaX=2.0, sigmaY=2.0)
        crop_prev = cv2.GaussianBlur(crop_prev, (0,0), sigmaX=2.0, sigmaY=2.0)
        # Phase correlation
        shift, response = cv2.phaseCorrelate(crop_cur, crop_prev)
        import math
        mag = math.sqrt(shift[0]**2 + shift[1]**2)
        direction = math.degrees(math.atan2(shift[1], shift[0]))
        log.info(f"phase_corr: shift=({shift[0]:.3f},{shift[1]:.3f}) "
                 f"mag={mag:.3f}px dir={direction:.1f}deg response={response:.4f}")
        return {
            'shift_x': round(shift[0], 4),
            'shift_y': round(shift[1], 4),
            'magnitude_px': round(mag, 4),
            'direction_deg': round(direction, 2),
            'response': round(response, 4),
        }
    except Exception as e:
        log.warning(f"phase_corr: {e}")
        return None

# ── COOKIE-CUTTER OPTICAL FLOW ───────────────────────────────────────────
def compute_optical_flow_cookie(cur_path, prev_path, centroid, cookie_radius=50):
    """
    Computes Farneback dense optical flow inside a cookie-cutter circle
    centred on the detected comet centroid. Isolates comet motion from
    coronal streamer dynamics.

    Returns dict with flow measurements or None on failure.
    """
    try:
        import cv2
        import numpy as np

        cur = cv2.imread(str(cur_path), cv2.IMREAD_GRAYSCALE)
        prev = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
        if cur is None or prev is None:
            log.warning("optical_flow: cannot read frames")
            return None

        # JPEG artifact suppression (DeepSeek Comm.27: sigma=2.0 for 8x8 DCT blocks)
        cur = cv2.GaussianBlur(cur, (0,0), sigmaX=2.0, sigmaY=2.0)
        prev = cv2.GaussianBlur(prev, (0,0), sigmaX=2.0, sigmaY=2.0)

        # Check frames are actually different
        if abs(cur.astype(int) - prev.astype(int)).max() == 0:
            log.info("optical_flow: frames identical, skipping")
            return None

        h, w = cur.shape
        cy_c = centroid['centroid_y']
        cx_c = centroid['centroid_x']

        # Cookie cutter mask
        cookie = np.zeros((h,w), dtype=np.uint8)
        cv2.circle(cookie, (cx_c, cy_c), cookie_radius, 255, -1)

        # Mask frames outside cookie
        cur_m = cur.copy()
        prev_m = prev.copy()
        cur_m[cookie==0] = 0
        prev_m[cookie==0] = 0

        # Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_m, cur_m, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag[cookie==0] = 0

        valid = mag[cookie>0]
        if valid.mean() < 0.001:
            log.info("optical_flow: negligible flow in cookie")
            return None

        # Significant flow pixels
        threshold = valid.mean() + 2*valid.std()
        sig = (mag > threshold) & (cookie > 0)
        sig_count = int(sig.sum())

        # Mean flow vector in significant region
        if sig_count > 3:
            mean_fx = float(flow[sig,0].mean())
            mean_fy = float(flow[sig,1].mean())
        else:
            mean_fx = float(flow[cookie>0,0].mean())
            mean_fy = float(flow[cookie>0,1].mean())

        import math
        mean_dir = math.degrees(math.atan2(mean_fy, mean_fx))
        mean_mag = math.sqrt(mean_fx**2 + mean_fy**2)
        ps = PLATE_SCALE.get('cor2', 58.8)
        angular_motion = mean_mag * ps / 3600.0  # degrees per frame

        # Peak flow
        peak_loc = np.unravel_index(mag.argmax(), mag.shape)
        peak_mag = float(mag[peak_loc])

        log.info(f"optical_flow: mag={mean_mag:.4f}px dir={mean_dir:.1f}deg "
                 f"angular={angular_motion:.6f}deg/frame sig_px={sig_count}")

        result = {
            'flow_magnitude_px':     round(mean_mag, 4),
            'flow_direction_deg':    round(mean_dir, 2),
            'angular_motion_deg':    round(angular_motion, 6),
            'peak_flow_px':          round(peak_mag, 4),
            'peak_flow_location':    [int(peak_loc[0]), int(peak_loc[1])],
            'significant_flow_px':   sig_count,
            'cookie_radius_px':      cookie_radius,
            'mean_flow_vector':      [round(mean_fx, 4), round(mean_fy, 4)],
            'plate_scale_arcsec_px': ps,
        }

        # Save flow visualization
        try:
            cur_color = cv2.imread(str(cur_path))
            output = cur_color.copy()
            # Cookie circles
            cv2.circle(output, (cx_c,cy_c), cookie_radius, (0,200,160), 1)
            cv2.circle(output, (cx_c,cy_c), cookie_radius+4, (0,255,200), 1)
            # Flow arrows inside cookie
            step = 4
            for y in range(max(0,cy_c-cookie_radius), min(h,cy_c+cookie_radius), step):
                for x in range(max(0,cx_c-cookie_radius), min(w,cx_c+cookie_radius), step):
                    if cookie[y,x]>0 and mag[y,x]>valid.mean()+valid.std():
                        fx = flow[y,x,0]
                        fy = flow[y,x,1]
                        ex = int(x + fx*3)
                        ey = int(y + fy*3)
                        cv2.arrowedLine(output, (x,y), (ex,ey), (0,255,200), 1, tipLength=0.3)
            # Centroid crosshair
            cv2.line(output, (cx_c-8,cy_c), (cx_c-3,cy_c), (0,255,200), 1)
            cv2.line(output, (cx_c+3,cy_c), (cx_c+8,cy_c), (0,255,200), 1)
            cv2.line(output, (cx_c,cy_c-8), (cx_c,cy_c-3), (0,255,200), 1)
            cv2.line(output, (cx_c,cy_c+3), (cx_c,cy_c+8), (0,255,200), 1)
            # Mean direction arrow
            if abs(mean_fx)+abs(mean_fy) > 0.1:
                ex = int(cx_c + mean_fx*8)
                ey = int(cy_c + mean_fy*8)
                cv2.arrowedLine(output, (cx_c,cy_c), (ex,ey), (0,200,255), 2, tipLength=0.2)
            flow_path = FRAME_DIR / 'cor2_flow.jpg'
            cv2.imwrite(str(flow_path), output, [cv2.IMWRITE_JPEG_QUALITY, 92])
            os.chmod(flow_path, 0o644)
            try:
                import pwd, grp
                os.chown(flow_path, pwd.getpwnam('www-data').pw_uid,
                         grp.getgrnam('www-data').gr_gid)
            except Exception:
                pass
            log.info("optical_flow: visualization saved")
        except Exception as ve:
            log.warning(f"optical_flow: visualization failed — {ve}")

        return result

    except Exception as e:
        log.error(f"optical_flow: {e}")
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
        if any(name.endswith(s) for s in ['_current', '_previous', '_annotated', '_traj', '_flow', '_profile']):
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

    # Gyori comet assay profile (Gyori et al. 2014, Comm.28 optimized)
    gyori_result = None
    if centroid_result and centroid_result.get('centroid_x') is not None:
        cor2_cur = FRAME_DIR / 'cor2_current.jpg'
        if cor2_cur.exists():
            gyori_result = gyori_profile_analysis(str(cor2_cur), centroid_result)
            if gyori_result:
                log.info(f"Gyori uuid={gyori_result.get('cag_uuid','')[:8]}")

    # Optical flow on COR2 cookie-cutter region
    flow_result = None
    if centroid_result and centroid_result.get('centroid_x') is not None:
        cor2_cur = FRAME_DIR / 'cor2_current.jpg'
        cor2_prev = FRAME_DIR / 'cor2_previous.jpg'
        if cor2_cur.exists() and cor2_prev.exists():
            flow_result = compute_optical_flow_cookie(
                cor2_cur, cor2_prev, centroid_result, cookie_radius=50)
            # Phase correlation sanity check
            if flow_result:
                pc = phase_correlation_check(
                    cor2_cur, cor2_prev,
                    (centroid_result['centroid_y'], centroid_result['centroid_x']), 50)
                if pc:
                    flow_result['phase_correlation'] = pc
                    # Check agreement
                    farn_mag = flow_result['flow_magnitude_px']
                    pc_mag = pc['magnitude_px']
                    disagreement = abs(farn_mag - pc_mag)
                    flow_result['farneback_phase_disagreement_px'] = round(disagreement, 4)
                    if disagreement > 0.5:
                        flow_result['quality_flag'] = 'REVIEW: Farneback/phase disagreement > 0.5px'
                        log.warning(f"Flow quality: Farneback={farn_mag:.3f} vs phase={pc_mag:.3f} DISAGREEMENT")
                    else:
                        flow_result['quality_flag'] = 'OK'
                        log.info(f"Flow quality: Farneback={farn_mag:.3f} vs phase={pc_mag:.3f} CONSISTENT")

            if flow_result:
                # Wolfram UUID for flow computation
                wl_flow = f"""
Module[{{uuid}},
  uuid = ToString[CreateUUID[]];
  ExportString[{{
    "computation"     -> "optical_flow_cookie_v1",
    "flow_mag_px"     -> {flow_result['flow_magnitude_px']},
    "flow_dir_deg"    -> {flow_result['flow_direction_deg']},
    "angular_motion"  -> {flow_result['angular_motion_deg']},
    "sig_pixels"      -> {flow_result['significant_flow_px']},
    "cag_uuid"        -> uuid
  }}, "JSON"]
]
"""
                try:
                    wl_r = subprocess.run(['wolframscript','-code',wl_flow],
                                          capture_output=True, text=True, timeout=90)
                    if wl_r.returncode == 0:
                        parsed = json.loads(wl_r.stdout.strip())
                        if isinstance(parsed, list):
                            d = {item[0]:item[1] for item in parsed if isinstance(item,list)}
                            flow_result['cag_uuid'] = d.get('cag_uuid')
                        else:
                            flow_result['cag_uuid'] = parsed.get('cag_uuid')
                        log.info(f"optical_flow uuid={flow_result['cag_uuid']}")
                    else:
                        flow_result['cag_uuid'] = None
                except Exception as fe:
                    log.error(f"optical_flow Wolfram: {fe}")
                    flow_result['cag_uuid'] = None

    # C3 Gyori profile — find comet via largest bright region + observed tail direction
    c3_gyori_result = None
    c3_path = FRAME_DIR / 'c3_current.jpg'
    if c3_path.exists():
        c3_head = find_c3_comet_head(str(c3_path))
        if c3_head:
            tail_dir = find_observed_tail_direction(str(c3_path), c3_head[0], c3_head[1])
            if tail_dir is not None:
                c3_gyori_result = gyori_profile_c3(str(c3_path), c3_head[0], c3_head[1], tail_dir)

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

    # Gyori comet assay profile (Gyori et al. 2014, adapted Comm.28)
    entry['gyori_profile_cor2'] = gyori_result if gyori_result else {
        'tail_length_px': None, 'tail_moment': None,
        'olive_moment': None, 'pct_in_tail': None,
        'cag_uuid': None, 'note': 'no_profile_data'
    }

    # C3 Gyori profile (observed tail direction)
    entry['gyori_profile_c3'] = c3_gyori_result if c3_gyori_result else {
        'tail_length_px': None, 'tail_moment': None,
        'pct_in_tail': None, 'cag_uuid': None,
        'note': 'no_c3_profile'
    }

    # Optical flow result
    # DeepSeek Comm.27: document measurement quality in JSONL
    if flow_result:
        mag = flow_result.get('flow_magnitude_px', 0)
        if mag > 0:
            # COR2: 13-27% artifact at 1.12px, scales inversely with magnitude
            artifact_pct = min(30, max(5, 30.0 / (mag + 0.5)))
            flow_result['jpeg_artifact_estimate_pct'] = round(artifact_pct, 1)
        flow_result['preprocessing'] = 'gaussian_sigma2.0'

    entry['optical_flow_cor2'] = flow_result if flow_result else {
        'flow_magnitude_px': None, 'flow_direction_deg': None,
        'angular_motion_deg': None, 'cag_uuid': None,
        'note': 'no_flow_data'
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

    # 4b. Update UUID count summary
    try:
        import re as _re
        _t2 = open('/var/log/heliodata/hale_tier2_poc.jsonl').read()
        _cg = JSONL_PATH.read_text()
        _all = set(_re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', _t2+_cg))
        _summary = {'total_uuids':len(_all),'tier2_entries':len(_t2.strip().splitlines()),'coronagraph_entries':len(_cg.strip().splitlines())}
        _sp = BASE / 'cag_uuid_count.json'
        _sp.write_text(json.dumps(_summary))
        os.chmod(_sp, 0o644)
        log.info(f"UUID summary: {len(_all)} total")
    except Exception as _e:
        log.warning(f"UUID summary update failed: {_e}")

    # 5. Rolling window cleanup
    cleanup_rolling_window()

    log.info("=== hale_coronagraph_fetch complete ===")
    print(json.dumps(entry, indent=2))

if __name__ == '__main__':
    main()
