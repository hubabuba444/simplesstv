"""
Simple SSTV decoder (Robot32, Scottie1, PD120) — improved sync handling, per‑mode timing, and Hilbert demod.

Usage:
    python sstv_decoder.py --input_wav input.wav --input_mode scottie1

Dependencies:
    pip install numpy scipy pygame

Notes:
 - Displays the image live in a pygame window and logs to console.
 - Designed for clarity & robustness over speed; works on partial frames.
 - Scottie1 uses the true R–G —sync— B line structure.
 - PD120 handles Y, (R‑Y), (B‑Y), Y with progressive 2‑line luminance.

"""

import argparse
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import pygame

# ------------------ Mode parameters (from published specs) ------------------
# Key refs you can look up: Scottie1 line: 1.5 ms porch, 138.240 ms R, 1.5 ms porch,
# 138.240 ms G, 9.0 ms sync, 1.5 ms porch, 138.240 ms B. Total line ~428.22 ms.
# PD120: per line: 20 ms sync, 2.08 ms porch, then Y (odd), R-Y, B-Y, Y (even),
# each 121.600 ms. Display 640x496.
# Robot32: B/W, ~32 s per 256x240 image ≈ 125 ms/line; we treat porch/sync generically.

MODES = {
    'robot36': {
    'name': 'Robot 36',
    'width': 320,
    'height': 240,
    'frame_time': 36.0,
    'components_per_line': 3,
    'sync_skip_s': 0.009,
    'porch_skip_s': 0.0015,
    'scan': 'RGB',
    'color_order': ['G','B','R']
    },

    'scottie1': {
        'name': 'Scottie 1',
        'width': 320,
        'height': 256,           # Scottie 1 is 320x256
        'sync_hz': 1200.0,
        'sync_ms': 9.0,
        'porch_ms': 1.5,         # “black” porch (1500 Hz)
        'scan_ms': 138.240,      # per-color
        'line_ms': 428.22,       # sync-to-sync (approx)
    },
    'pd120': {
        'name': 'PD120',
        'width': 640,
        'height': 496,           # includes 16-line header
        'sync_hz': 1200.0,
        'sync_ms': 20.0,
        'porch_ms': 2.08,
        'scan_ms': 121.600,      # Y, R-Y, B-Y, Y
        'line_ms': 20.0 + 2.08 + 4*121.6,
    }
}

# Core SSTV tone mapping
FREQ_MIN = 1500.0
FREQ_MAX = 2300.0
FREQ_MID = 1900.0
SYNC_THRESH = 1350.0  # Hz; anything below is likely sync

# ------------------ Signal helpers ------------------

def bandpass(x, sr, lo=900.0, hi=2600.0, order=6):
    b, a = sig.butter(order, [lo/(sr/2), hi/(sr/2)], btype='band')
    return sig.filtfilt(b, a, x)


def inst_freq_hz(x, sr):
    """Instantaneous frequency (Hz) via Hilbert transform, smoothed a bit."""
    z = sig.hilbert(x)
    phase = np.unwrap(np.angle(z))
    fi = (sr/(2.0*np.pi)) * np.diff(phase)
    # pad to length x
    fi = np.concatenate([fi, fi[-1:]])
    # smooth ~1.2 ms moving average
    win = int(max(3, sr * 0.0012))
    if win > 3:
        fi = sig.convolve(fi, np.ones(win)/win, mode='same')
    return fi


def hz_to_luma(fi):
    # Map 1500..2300 Hz -> 0..1
    y = (fi - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)
    return np.clip(y, 0.0, 1.0)


def hz_to_chroma(fi):
    # Map around 1900 Hz center -> [-1..1]
    c = (fi - FREQ_MID) / ((FREQ_MAX - FREQ_MIN)/2.0)
    return np.clip(c, -1.0, 1.0)


def sample_segment(values, start_idx, dur_s, n_out, kind='luma'):
    """Sample instantaneous-frequency-derived values evenly over a time window."""
    N = len(values)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    stop_idx = start_idx + int(round(dur_s))
    stop_idx = max(min(stop_idx, N-1), start_idx+1)
    idxs = np.linspace(start_idx, stop_idx-1, n_out)
    vi = np.interp(idxs, np.arange(N), values)
    if kind == 'luma':
        return hz_to_luma(vi)
    else:
        return hz_to_chroma(vi)


def detect_sync_indices(fi, sr, sync_ms, expected_line_ms):
    """Detect start indices of line sync pulses (1200 Hz)."""
    is_sync = fi < SYNC_THRESH
    # find start of low segments
    edges = np.flatnonzero((~is_sync[:-1]) & (is_sync[1:])) + 1
    sync_idxs = []
    min_len = int(sr * (max(0.6*sync_ms, 4.0)/1000.0))
    max_len = int(sr * (min(1.6*sync_ms, 30.0)/1000.0))
    for s in edges:
        e = s
        while e < len(is_sync) and is_sync[e]:
            e += 1
        seg_len = e - s
        if seg_len >= min_len and seg_len <= max_len:
            sync_idxs.append(s)
    sync_idxs = np.array(sync_idxs, dtype=np.int64)
    if len(sync_idxs) < 4:
        return sync_idxs
    # Prune outliers based on line spacing
    diffs = np.diff(sync_idxs)
    est = int(sr * (expected_line_ms/1000.0))
    ok = (diffs > 0.5*est) & (diffs < 1.5*est)
    keep = np.where(np.concatenate([[True], ok]))[0]
    return sync_idxs[keep]

# ------------------ Decoders ------------------

def decode_scottie1(fi, sr):
    P = MODES['scottie1']
    W, H = P['width'], P['height']
    sync_s = P['sync_ms']/1000.0
    porch_s = P['porch_ms']/1000.0
    scan_s = P['scan_ms']/1000.0

    sync_idx = detect_sync_indices(fi, sr, P['sync_ms'], P['line_ms'])
    print(f"[INFO] Scottie1: found {len(sync_idx)} syncs")

    # Output buffers
    R = np.zeros((H, W), dtype=np.float32)
    G = np.zeros((H, W), dtype=np.float32)
    B = np.zeros((H, W), dtype=np.float32)

    # Working index for image line
    line_i = 0
    # We will place: for interval [sync[i] .. sync[i+1]):
    #   Blue for line_i (current), then Red+Green for line_i+1
    for i in range(len(sync_idx)-1):
        s0 = sync_idx[i]
        # Segment layout starting right AFTER sync pulse
        t = s0 + int(sr*sync_s)  # end of sync
        # Blue (current line_i)
        start_B = t + int(sr*porch_s)
        B_line = sample_segment(fi, start_B, int(sr*scan_s), W, kind='chroma')  # actually luma mapping for RGB channels
        # For Scottie, all three are direct color channels (not R-Y/B-Y), so we should use luma mapping for them
        B_line = sample_segment(fi, start_B, sr*scan_s, W, kind='luma')

        # After blue, porch then Red (for next line)
        after_B = start_B + int(sr*scan_s)
        start_R = after_B + int(sr*porch_s)
        R_next = sample_segment(fi, start_R, sr*scan_s, W, kind='luma')

        # After red, porch then Green (for next line)
        after_R = start_R + int(sr*scan_s)
        start_G = after_R + int(sr*porch_s)
        G_next = sample_segment(fi, start_G, sr*scan_s, W, kind='luma')

        if line_i < H:
            B[line_i, :] = B_line
        if line_i+1 < H:
            R[line_i+1, :] = R_next
            G[line_i+1, :] = G_next
        line_i += 1

    # Fill the missing channels on first/last lines by simple propagation
    if H >= 2:
        R[0, :] = R[1, :]
        G[0, :] = G[1, :]
        B[-1, :] = B[-2, :]

    img = np.clip(np.stack([R, G, B], axis=-1) * 255.0, 0, 255).astype(np.uint8)
    used_lines = min(line_i+1, H)
    img = img[:used_lines, :, :]
    return img


def decode_pd120(fi, sr):
    P = MODES['pd120']
    W, H = P['width'], P['height']
    sync_s = P['sync_ms']/1000.0
    porch_s = P['porch_ms']/1000.0
    scan_s = P['scan_ms']/1000.0

    sync_idx = detect_sync_indices(fi, sr, P['sync_ms'], P['line_ms'])
    print(f"[INFO] PD120: found {len(sync_idx)} syncs")

    R = np.zeros((H, W), dtype=np.float32)
    G = np.zeros((H, W), dtype=np.float32)
    B = np.zeros((H, W), dtype=np.float32)

    line = 0
    for i in range(len(sync_idx)-1):
        s0 = sync_idx[i]
        t = s0 + int(sr*sync_s)
        # After porch
        t += int(sr*porch_s)

        # Y (odd line)
        Y0 = sample_segment(fi, t, sr*scan_s, W, kind='luma')
        t += int(sr*scan_s)
        # R-Y and B-Y (shared)
        V = sample_segment(fi, t, sr*scan_s, W, kind='chroma')  # R-Y
        t += int(sr*scan_s)
        U = sample_segment(fi, t, sr*scan_s, W, kind='chroma')  # B-Y
        t += int(sr*scan_s)
        # Y (even line)
        Y1 = sample_segment(fi, t, sr*scan_s, W, kind='luma')

        # Convert to RGB; use standard SDTV-ish matrix:
        # R = Y + (R-Y) ; B = Y + (B-Y)
        # G = Y - 0.509*(R-Y) - 0.194*(B-Y)
        if line < H:
            R[line, :] = np.clip(Y0 + V, 0.0, 1.0)
            B[line, :] = np.clip(Y0 + U, 0.0, 1.0)
            G[line, :] = np.clip(Y0 - 0.509*V - 0.194*U, 0.0, 1.0)
        if line+1 < H:
            R[line+1, :] = np.clip(Y1 + V, 0.0, 1.0)
            B[line+1, :] = np.clip(Y1 + U, 0.0, 1.0)
            G[line+1, :] = np.clip(Y1 - 0.509*V - 0.194*U, 0.0, 1.0)
        line += 2

    img = np.clip(np.stack([R, G, B], axis=-1) * 255.0, 0, 255).astype(np.uint8)
    used_lines = min(line, H)
    img = img[:used_lines, :, :]
    return img


def decode_robot32(fi, sr):
    P = MODES['robot32']
    W, H = P['width'], P['height']
    sync_idx = detect_sync_indices(fi, sr, P['sync_ms'], P['line_ms'])
    print(f"[INFO] Robot32: found {len(sync_idx)} syncs")

    Y = np.zeros((H, W), dtype=np.float32)
    for i in range(min(H, len(sync_idx)-1)):
        s0 = sync_idx[i]
        # Start after sync and porch
        t = s0 + int(sr*(P['sync_ms']/1000.0)) + int(sr*(P['porch_ms']/1000.0))
        # Scan duration: use line_ms minus overhead as approximation
        scan_ms = max(40.0, P['line_ms'] - P['sync_ms'] - P['porch_ms'])
        yline = sample_segment(fi, t, sr*(scan_ms/1000.0), W, kind='luma')
        Y[i, :] = yline

    img = np.clip(np.stack([Y, Y, Y], axis=-1) * 255.0, 0, 255).astype(np.uint8)
    used_lines = min(H, len(sync_idx)-1)
    img = img[:used_lines, :, :]
    return img

# ------------------ I/O & main ------------------

def load_wav(path):
    sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    # normalize
    if np.issubdtype(audio.dtype, np.integer):
        maxv = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / maxv
    else:
        audio = audio.astype(np.float32)
    return sr, audio


def decode_from_audio(sr, audio, mode):
    mode = mode.lower()
    if mode not in MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Use one of: {list(MODES.keys())}")

    P = MODES[mode]
    print(f"[INFO] Read WAV: sample_rate={sr}  shape={audio.shape}")
    print(f"[INFO] Decoding mode {P['name']}  {P['width']}x{P['height']}  est line={P['line_ms']:.1f} ms")

    # Preprocess: bandpass and instantaneous frequency
    x = bandpass(audio, sr)
    fi = inst_freq_hz(x, sr)
    print(f"[INFO] Inst. frequency computed  len={len(fi)} samples")

    if mode == 'scottie1':
        return decode_scottie1(fi, sr)
    elif mode == 'pd120':
        return decode_pd120(fi, sr)
    else:
        return decode_robot32(fi, sr)


def show_pygame(img, title='SSTV decode'):
    h, w = img.shape[:2]
    pygame.init()
    surface = pygame.display.set_mode((w, h))
    pygame.display.set_caption(title)
    # Convert to pygame surface (RGB)
    surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    surface.blit(surf, (0, 0))
    pygame.display.flip()
    print('[INFO] Displaying image — close the window to exit')
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_wav', required=True, help='Path to WAV file with SSTV audio')
    ap.add_argument('--input_mode', required=True, choices=list(MODES.keys()), help='Mode: robot36 | scottie1 | pd120')
    args = ap.parse_args()

    sr, audio = load_wav(args.input_wav)
    img = decode_from_audio(sr, audio, args.input_mode)
    show_pygame(img, f"{MODES[args.input_mode]['name']} — {img.shape[1]}x{img.shape[0]}")

if __name__ == '__main__':
    main()
