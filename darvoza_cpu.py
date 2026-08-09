"""
Employee Attendance Tracker - Multi-Camera Check-In / Check-Out
------------------------------------------------------------------
Watches ANY NUMBER of camera feeds at once (one worker thread per camera),
detects people with YOLO, and identifies faces against a folder of known
employee photos using InsightFace (ArcFace embeddings).

Each camera has a ROLE, which decides what kind of Telegram notification
a recognized employee triggers on it:
  - "entry"      -> seeing them here means they ENTERED. Sends an "entered"
                    notification and starts their attendance clock.
  - "exit"       -> seeing them here means they LEFT. Sends a "left"
                    notification with how long they were present.
  - "entry_exit" -> one camera watches a single door people both enter AND
                     leave through. Seeing them here TOGGLES their state:
                     not checked in yet -> "entered", already checked in ->
                     "left". Debounced (ENTRY_EXIT_TOGGLE_DEBOUNCE_SECONDS)
                     so a brief tracking hiccup can't flip someone twice.
  - "interior"   -> any other camera. Seeing them here just means they were
                    spotted somewhere in the building - sends a lightweight
                    "appeared on camera" notification (throttled, so standing
                    in view doesn't spam you every few seconds).

ACCURACY, for different face angles:
  - Each employee's photos are turned into a GALLERY of embeddings (one per
    photo, PLUS a horizontally-mirrored version of each) instead of a single
    averaged template. Averaging embeddings from very different angles into
    one vector tends to wash out the details that make matching accurate;
    comparing a live face against the closest entry in a gallery holds onto
    each angle's detail instead. Feeding it a few photos per employee from
    different angles (front, both 3/4 profiles) matters more for accuracy
    than anything else in this file - see README.md.
  - A live identification only "locks on" to a name after
    IDENTITY_LOCK_CONSECUTIVE_READS separate camera reads agree on the same
    person, so one bad-angle or backlit frame can't misidentify someone.
  - A face match must also beat the SECOND-closest employee by at least
    MIN_MATCH_MARGIN, not just clear the threshold. This is what stops two
    similar-looking employees from being confused with each other - a match
    that's only barely closer to one gallery than another is treated as
    "unclear" and simply skipped rather than guessed at.
  - DAILY CLOTHING ASSIST: every time a face confidently confirms someone,
    the script also learns a rough color profile of their clothing for
    that calendar day (see learn_clothing()). If a later read can't get a
    confident face match (bad angle, face turned away, backlit, etc.) but
    the person's clothing unambiguously matches exactly one employee's
    already-learned outfit for today, that's used as a weaker secondary
    vote. It requires more agreeing reads (CLOTHING_LOCK_CONSECUTIVE_READS)
    than a face match does, and it only ever reinforces an identity that a
    face already established earlier that day - it never originates one
    from scratch, since clothes (unlike faces) aren't unique to a person.
    The learned outfit is reset automatically every day.

See README.md for setup instructions.
"""

import os
import csv
import time
import pickle
import logging
import threading
import traceback
import datetime

import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis
from ultralytics import YOLO

logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# Force RTSP to use TCP instead of the default UDP - avoids a lot of the random
# frame drops / disconnects IP cameras are prone to over UDP. Must be set before
# any cv2.VideoCapture(...) call.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|max_delay;500000",
)

# ============================================================
# SETTINGS - edit these before running
# ============================================================

# --- Cameras: add one entry per camera. There is no limit on how many. ---s
#
# role: "entry"      - being recognized here = "entered" notification
#       "exit"        - being recognized here = "left" notification
#       "entry_exit"  - one camera watching a door people both enter AND
#                        leave through. Being recognized here TOGGLES them:
#                        not checked in yet -> "entered", already checked in
#                        -> "left". See ENTRY_EXIT_TOGGLE_DEBOUNCE_SECONDS.
#       "interior"    - being recognized here = "appeared on camera" notification
#       (default is "interior" if you omit the field)
#
# Credentials are read from environment variables instead of being hardcoded
# in this file. Set them once per machine, e.g.:
#   export CAM1_USER=admin
#   export CAM1_PASS=your-camera-password
CAMERAS = [
    {
        "label": "Entrance",
        # Single camera watching the one door people both enter and leave
        # through - toggles clock-in/clock-out instead of only one direction.
        "role": "entry_exit",
        # Read from an env var so the real camera password never sits in this
        # file - set before running: export CAMERA_ENTRANCE_RTSP="YOUR_RTSP"
        "source": os.environ.get("CAMERA_ENTRANCE_RTSP", "YOUR_RTSP"),
        # Rectangles (x1, y1, x2, y2) in pixel coordinates to EXCLUDE from person
        # detection entirely - e.g. the street/sidewalk visible through a glass
        # storefront door, where passersby who never entered the store would
        # otherwise get detected, tracked, and run through face recognition for
        # no reason (wasted compute, and noise in your attendance data). Set
        # DEBUG_PRINT_BOX_COORDS = True further down to see live coordinates as
        # people move through frame, which makes it easy to figure out the right
        # rectangle for your camera's exact view. Leave as [] if not needed.
        "ignore_zones": [],
    },
    # Add more cameras here later - copy the block above, give it a new
    # env var name (e.g. CAMERA_KASSA_RTSP), and set "label"/"role" ("entry" /
    # "exit" / "entry_exit" / "interior").
]

# Folder containing one subfolder per employee, each with a few clear face
# photos - ideally from a few different angles (see README.md).
EMPLOYEE_PHOTOS_FOLDER = "employees"
EMPLOYEE_CACHE_FILE = "employee_face_cache.pkl"

# Telegram bot credentials - read from environment variables, never hardcoded.
#   export ATTENDANCE_TG_TOKEN=123456:ABC-your-bot-token
#   export ATTENDANCE_TG_CHAT_ID=123456789
TELEGRAM_BOT_TOKEN = os.environ.get("ATTENDANCE_TG_TOKEN", "YOUR_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("ATTENDANCE_TG_CHAT_ID", "YOUR_ID")

# Where attendance events are logged: one row per entered/left/appeared event.
ATTENDANCE_CSV_PATH = "attendance_log.csv"

# Send a snapshot photo along with "entered" and "appeared" Telegram messages
# (lets you visually confirm who was recognized). "Left" messages are text-only.
SEND_PHOTO_ON_NOTIFICATION = True

# --- Timing ---

# How often (in seconds) each camera runs YOLO + face recognition. This is
# the main lever for "how fast can a brand-new person even get their first
# read" - everything below is gated behind this cadence. 0.3-0.4 is a solid
# GPU default; if you're CPU-only and see the console falling behind (frames
# queuing up, detection lagging noticeably behind the live camera), raise
# this back toward 0.8-1.0 rather than let it silently backlog.
CHECK_EVERY_SECONDS = 0.3

# Once a person is being tracked locally on a camera, how often (seconds) to
# re-run face recognition on them, until they're locked on to a name (see
# IDENTITY_LOCK_CONSECUTIVE_READS below). IMPORTANT: this ONLY affects people
# who are NOT YET identified - once a track locks onto a name, it is never
# re-checked again for as long as that track exists, so lowering this has NO
# effect on steady-state load for already-recognized people. It only
# controls how quickly a BRAND NEW person's identity gets confirmed. 0.4-0.5
# still gives IDENTITY_LOCK_CONSECUTIVE_READS genuinely distinct frames to
# agree on (the person has time to shift slightly between reads), it's just
# not artificially throttled to a multi-second gap for no accuracy benefit.
REIDENTIFY_INTERVAL_SECONDS = 0.4

# A live identification only becomes trusted once this many separate reads
# in a row agree on the same employee. Raise this (e.g. to 3) if you're
# still seeing occasional wrong-name flickers; 2 is a reasonable default.
IDENTITY_LOCK_CONSECUTIVE_READS = 2

# Minimum time (seconds) between two "appeared on camera" (interior-role)
# notifications for the SAME employee on the SAME camera - stops it spamming
# you every REIDENTIFY_INTERVAL_SECONDS while someone just stands in view.
APPEARED_NOTIFICATION_COOLDOWN_SECONDS = 30

# For a camera with role "entry_exit" (one camera watching a door people both
# enter AND leave through - see CAMERAS below): being recognized TOGGLES their
# clock-in/clock-out state instead of only ever meaning one direction. This is
# the minimum time (seconds) that must pass since their last toggle before a
# new sighting is allowed to toggle them again. It exists purely to stop a
# brief tracking hiccup (the camera losing and re-acquiring the same person
# for a moment while they're still standing in the doorway) from flipping
# them in and back out again. It is NOT meant to model how long someone
# should take to leave - keep it short.
ENTRY_EXIT_TOGGLE_DEBOUNCE_SECONDS = 30

# Safety-net only: if someone is marked "entered" but is never seen on an
# EXIT-role camera, auto-mark them checked-out after this long so they don't
# stay "present" forever (e.g. they left through a door with no camera, or
# your exit camera missed them). Not used at all if every entered employee
# eventually passes an exit camera. Default: 8 hours.
CHECKOUT_FALLBACK_SECONDS = 8 * 60 * 60
CHECKOUT_SWEEP_INTERVAL_SECONDS = 60

# How many camera reads can fail in a row before that camera's connection is
# treated as truly lost and reconnected.
MAX_CONSECUTIVE_READ_FAILURES = 5

# If no new frame arrives for this many seconds, force a reconnect even if
# cap.read() hasn't technically returned a failure yet (some cameras stall or
# repeat a stale frame instead of cleanly erroring out).
FRAME_STALE_SECONDS = 8

# How many seconds a single connection ATTEMPT may hang before giving up and
# retrying, so a camera with a stalled handshake can't freeze its thread
# forever on the very first attempt.
CAMERA_OPEN_TIMEOUT_SECONDS = 10

# --- Detection / recognition thresholds ---

YOLO_MODEL_PATH = "yolo26s.pt"
PERSON_CLASS_ID = 0  # COCO class id for "person"
PERSON_CONFIDENCE = 0.6

# When True, prints every detected person's box coordinates to the console each
# detection cycle - turn this on temporarily to figure out what rectangle to
# put in a camera's "ignore_zones" (e.g. to exclude a street view visible
# through a glass door). Turn back off once you've got the numbers you need -
# it's noisy and adds console overhead you don't want running long-term.
DEBUG_PRINT_BOX_COORDS = False


def in_ignore_zone(x1, y1, x2, y2, zones):
    """True if this box's center falls inside any of the given ignore-zone
    rectangles [(zx1, zy1, zx2, zy2), ...]. Boxes inside a zone are skipped
    entirely - never tracked, never face-recognized."""
    if not zones:
        return False
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for (zx1, zy1, zx2, zy2) in zones:
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            return True
    return False


# Local per-camera tracking uses Ultralytics' bundled ByteTrack (motion-based,
# no extra model download required) - just needs "is this the same box as a
# moment ago on THIS camera", which ByteTrack handles well out of the box.
TRACKER_CONFIG = "bytetrack.yaml"

# InsightFace detector input resolution. Larger catches smaller/angled/
# farther-away faces more reliably (directly helps accuracy on off-angle
# faces) at the cost of a bit more compute per frame. 640x640 is the
# lightweight default; bump to (800, 800) or (960, 960) if accuracy matters
# more to you than raw speed, especially on GPU.
FACE_DET_SIZE = (800, 800)

# Minimum face size (pixels, shorter side) and blur threshold (variance of
# Laplacian - higher = sharper) for a face to be used at all, whether for
# building employee galleries or live recognition.
MIN_FACE_SIZE_PX = 36
FACE_BLUR_THRESHOLD = 55

# Employee reference photos are curated/known-good, so they get a looser
# quality bar on a second attempt if the strict pass finds nothing usable.
EMPLOYEE_MIN_FACE_SIZE_PX = 24
EMPLOYEE_FACE_BLUR_THRESHOLD = 25
EMPLOYEE_PHOTO_MAX_SIDE_PX = 1600

# Also add a horizontally-mirrored embedding for every employee reference
# photo. Doubles gallery coverage for free (a photo of someone's left side
# becomes usable for matching their right side too) - meaningfully helps
# recognizing off-angle/profile faces without needing more source photos.
EMPLOYEE_USE_FLIP_AUGMENTATION = True

# Cap on how many gallery embeddings one employee can have (photos x2 for
# flip augmentation). Keeps matching fast even with a lot of reference
# photos; raise if you have many photos per employee and want them all used.
EMPLOYEE_GALLERY_MAX_PER_PERSON = 60

# How close (cosine distance, LOWER = more similar) a live face embedding
# must be to the CLOSEST embedding in an employee's gallery to count as a
# match. 0.35-0.40 is a reasonable starting point for ArcFace; tighten it
# (lower) if you get false-positive matches between different employees,
# loosen it (higher) if a real employee isn't being recognized.
EMPLOYEE_MATCH_THRESHOLD = 0.38

# On top of clearing EMPLOYEE_MATCH_THRESHOLD, the best-matching employee must
# beat the SECOND-best-matching employee's distance by at least this much.
# Without this, two employees who happen to look somewhat alike can "steal"
# each other's reads whenever one gallery is randomly a hair closer than the
# other on a given frame. Raise it if you still see occasional swapped names
# between two specific people; lower it if genuine matches are being thrown
# out as "too ambiguous" (check the console - it prints when this happens).
MIN_MATCH_MARGIN = 0.04

# --- Daily clothing assist (secondary signal, never used on its own) ---
#
# Every time a face confidently identifies someone, the script also samples
# a color histogram of their clothing (torso area of the person crop) and
# keeps a running-average "today's outfit" profile per employee. It's wiped
# and rebuilt from scratch every calendar day.
#
# When a live read's face is inconclusive (no usable face angle, or a face
# match too close to call between two people), the script compares that
# read's clothing against every employee's outfit profile for TODAY ONLY. If
# exactly one employee is a clear, unambiguous match, that counts as a vote
# for their identity - but a weaker one than a face vote, so it needs more
# consecutive agreeing reads (CLOTHING_LOCK_CONSECUTIVE_READS) before a track
# is allowed to lock onto it. This mainly helps keep someone correctly
# identified while they turn away from the camera or walk past at an angle,
# after a face read already confirmed who they are earlier that same day.
CLOTHING_ASSIST_ENABLED = True
CLOTHING_HIST_BINS = (30, 32)          # (hue bins, saturation bins)
CLOTHING_MATCH_THRESHOLD = 0.25        # lower = stricter clothing match
CLOTHING_MATCH_MARGIN = 0.10           # best must beat 2nd-best employee by this much
CLOTHING_LEARN_SAMPLES_PER_DAY = 40    # running-average cap, keeps profile adaptive
CLOTHING_LOCK_CONSECUTIVE_READS = 4    # clothing is weaker evidence than a face - needs more reads

# This build is CPU-only - no GPU chooser, no CUDA path. YOLO_DEVICE="cpu"
# tells Ultralytics to run on CPU; INSIGHTFACE_CTX_ID=-1 tells InsightFace to
# run on CPU too (0 or higher would mean "use GPU index N").
YOLO_DEVICE = "cpu"
INSIGHTFACE_CTX_ID = -1


# Box colors (BGR) for the optional debug preview window
BOX_COLOR_KNOWN = (0, 200, 0)
BOX_COLOR_UNKNOWN = (0, 200, 255)
LABEL_TEXT_COLOR = (255, 255, 255)
BOX_THICKNESS = 2

# Pop up a live window per camera showing detection/recognition boxes in
# real time. Handy for setup/debugging; turn off for a truly headless run
# (e.g. running as a background service with no display attached).
SHOW_LIVE_PREVIEW = False

# Cap on how often the preview window actually redraws (frames per second).
# Without this, the preview redraws on EVERY frame the camera delivers (often
# 25+ fps) - all pure CPU spent on imshow/GUI overhead that has nothing to do
# with detection accuracy, and on a CPU-only machine it directly steals cycles
# from YOLO/InsightFace, which are also fighting for CPU. 12-15 is plenty
# smooth for a debug/monitoring window. Raise it if you want a snappier
# preview and have CPU to spare; lower it (or set SHOW_LIVE_PREVIEW = False)
# if you're still CPU-bound with multiple cameras open.
PREVIEW_MAX_FPS = 15

# Print WHY a face wasn't matched to an employee (too small, too blurry, no
# gallery built, distance too high, ambiguous between two people, etc.)
# instead of failing silently. Turn off once things are working reliably -
# it's noisy since it prints on every re-identify attempt.
DEBUG_FACE_MATCHING = True

VALID_ROLES = ("entry", "exit", "interior", "entry_exit")

# Set on Ctrl+C so camera worker threads can notice, close their OWN preview
# window (a window must be destroyed by the thread that created it - doing
# it from another thread is what leaves a dead, unclosable "Not Responding"
# window behind), and exit cleanly instead of being abruptly killed.
stop_event = threading.Event()

# ============================================================
# You shouldn't need to edit anything below this line
# ============================================================

face_app = None
face_engine_lock = threading.Lock()

employee_gallery = {}  # name -> list of unit-normalized embeddings (multiple angles)

employee_clothing_lock = threading.Lock()
employee_clothing = {}  # name -> {"date": "YYYY-MM-DD", "hist": np.ndarray, "samples": int}

# Building-wide attendance state, guarded by attendance_lock since every
# camera thread touches it.
attendance_lock = threading.Lock()
attendance_state = {}
# name -> {
#   "checked_in": bool,
#   "check_in_time": float | None,
#   "last_seen": float,
#   "last_camera": str,
#   "last_appeared": {camera_label: float},   # for interior-role cooldown
# }

csv_lock = threading.Lock()


# ---------------------------------------------------------------- Telegram --

def send_telegram_text(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("NOTE: Telegram not configured (ATTENDANCE_TG_TOKEN / ATTENDANCE_TG_CHAT_ID "
              "not set) - skipping notification.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
        if resp.status_code != 200:
            print(f"Telegram send failed: {resp.text}")
    except Exception as e:
        print(f"Could not send Telegram message: {e}")


def send_telegram_photo(image_bytes, caption):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        resp = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
            files={"photo": ("snapshot.jpg", image_bytes)},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"Telegram photo send failed: {resp.text}")
    except Exception as e:
        print(f"Could not send Telegram photo: {e}")


def notify(message, frame_for_photo=None):
    if SEND_PHOTO_ON_NOTIFICATION and frame_for_photo is not None:
        success, buffer = cv2.imencode(".jpg", frame_for_photo)
        if success:
            send_telegram_photo(buffer.tobytes(), message)
            return
    send_telegram_text(message)


# --------------------------------------------------------------- Attendance --

def format_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def append_attendance_row(name, event, camera_label, duration_str=""):
    file_exists = os.path.exists(ATTENDANCE_CSV_PATH)
    with csv_lock:
        with open(ATTENDANCE_CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "employee", "event", "camera", "duration"])
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, name, event, camera_label, duration_str])


def _get_or_init_state(name, now):
    state = attendance_state.get(name)
    if state is None:
        state = {
            "checked_in": False,
            "check_in_time": None,
            "last_seen": now,
            "last_camera": None,
            "last_appeared": {},
            "last_toggle": 0,  # for entry_exit role debounce - see ENTRY_EXIT_TOGGLE_DEBOUNCE_SECONDS
        }
        attendance_state[name] = state
    return state


def handle_sighting(name, camera_label, role, now, frame_for_photo=None):
    """Called whenever an employee is confidently recognized on any camera.
    What happens depends on that camera's role - see the module docstring."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if role == "entry":
        do_notify, message = False, None
        with attendance_lock:
            state = _get_or_init_state(name, now)
            state["last_seen"] = now
            state["last_camera"] = camera_label
            if not state["checked_in"]:
                state["checked_in"] = True
                state["check_in_time"] = now
                do_notify, message = True, (
                    f"\u2705 {name} ENTERED\nCamera: {camera_label}\nClock-in: {timestamp}"
                )
        if do_notify:
            print(f"[ENTERED] {name} on {camera_label} at {timestamp}")
            append_attendance_row(name, "entered", camera_label)
            notify(message, frame_for_photo)

    elif role == "exit":
        do_notify, message, duration_str = False, None, ""
        with attendance_lock:
            state = _get_or_init_state(name, now)
            state["last_seen"] = now
            state["last_camera"] = camera_label
            if state["checked_in"]:
                check_in_epoch = state["check_in_time"] or now
                duration = now - check_in_epoch
                duration_str = format_duration(duration)
                check_in_str = datetime.datetime.fromtimestamp(check_in_epoch).strftime("%Y-%m-%d %H:%M:%S")
                state["checked_in"] = False
                state["check_in_time"] = None
                do_notify, message = True, (
                    f"\U0001f6aa {name} LEFT\nCamera: {camera_label}\n"
                    f"Clock-in: {check_in_str}\nClock-out: {timestamp}\n"
                    f"Hours inside: {duration_str}"
                )
        if do_notify:
            print(f"[LEFT] {name} on {camera_label} at {timestamp} - present for {duration_str}")
            append_attendance_row(name, "left", camera_label, duration_str)
            notify(message, frame_for_photo=None)  # text-only for "left"

    elif role == "entry_exit":
        # One camera watches a door people both enter AND leave through, so a
        # recognition here doesn't inherently mean "entered" or "left" - it
        # means "toggle whatever they currently are". Debounced so a brief
        # tracking hiccup can't flip someone in and back out again while
        # they're still standing in the doorway - see
        # ENTRY_EXIT_TOGGLE_DEBOUNCE_SECONDS.
        do_notify, message, event, duration_str = False, None, None, ""
        with attendance_lock:
            state = _get_or_init_state(name, now)
            state["last_seen"] = now
            state["last_camera"] = camera_label
            if now - state["last_toggle"] >= ENTRY_EXIT_TOGGLE_DEBOUNCE_SECONDS:
                state["last_toggle"] = now
                if not state["checked_in"]:
                    state["checked_in"] = True
                    state["check_in_time"] = now
                    do_notify, event = True, "entered"
                    message = f"\u2705 {name} ENTERED\nCamera: {camera_label}\nClock-in: {timestamp}"
                else:
                    check_in_epoch = state["check_in_time"] or now
                    duration = now - check_in_epoch
                    duration_str = format_duration(duration)
                    check_in_str = datetime.datetime.fromtimestamp(check_in_epoch).strftime("%Y-%m-%d %H:%M:%S")
                    state["checked_in"] = False
                    state["check_in_time"] = None
                    do_notify, event = True, "left"
                    message = (
                        f"\U0001f6aa {name} LEFT\nCamera: {camera_label}\n"
                        f"Clock-in: {check_in_str}\nClock-out: {timestamp}\n"
                        f"Hours inside: {duration_str}"
                    )
        if do_notify:
            print(f"[{event.upper()}] {name} on {camera_label} at {timestamp}")
            append_attendance_row(name, event, camera_label, duration_str)
            notify(message, frame_for_photo if event == "entered" else None)

    else:  # "interior" - just note they were spotted, throttled
        do_notify = False
        with attendance_lock:
            state = _get_or_init_state(name, now)
            state["last_seen"] = now
            state["last_camera"] = camera_label
            last = state["last_appeared"].get(camera_label, 0)
            if now - last >= APPEARED_NOTIFICATION_COOLDOWN_SECONDS:
                state["last_appeared"][camera_label] = now
                do_notify = True
        if do_notify:
            print(f"[SEEN] {name} on {camera_label} at {timestamp}")
            append_attendance_row(name, "seen", camera_label)
            message = f"\U0001f440 {name} appeared on camera\nCamera: {camera_label}\nTime: {timestamp}"
            notify(message, frame_for_photo)


def checkout_fallback_sweep_loop():
    """Safety net only - see CHECKOUT_FALLBACK_SECONDS. Normal check-outs
    happen instantly via handle_sighting() on an exit-role camera; this just
    prevents someone from staying "checked in" forever if they never pass one."""
    while True:
        time.sleep(CHECKOUT_SWEEP_INTERVAL_SECONDS)
        now = time.time()
        to_checkout = []
        with attendance_lock:
            for name, state in attendance_state.items():
                if state["checked_in"] and now - state["last_seen"] >= CHECKOUT_FALLBACK_SECONDS:
                    to_checkout.append((name, dict(state)))
                    state["checked_in"] = False
                    state["check_in_time"] = None

        for name, state in to_checkout:
            duration = state["last_seen"] - state["check_in_time"]
            duration_str = format_duration(duration)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            check_in_str = datetime.datetime.fromtimestamp(state["check_in_time"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[LEFT - fallback] {name} (last seen on {state['last_camera']}) at {timestamp}")
            append_attendance_row(name, "left (auto, no exit camera sighting)", state["last_camera"], duration_str)
            message = (f"\U0001f6aa {name} auto-checked OUT (no exit camera sighting)\n"
                       f"Last seen: {state['last_camera']}\n"
                       f"Clock-in: {check_in_str}\nClock-out (approx): {timestamp}\n"
                       f"Hours inside (approx): {duration_str}")
            send_telegram_text(message)


# ------------------------------------------------------------- Face matching --

def init_face_engine():
    global face_app
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=FACE_DET_SIZE)
    print(f"InsightFace (buffalo_l) engine loaded, det_size={FACE_DET_SIZE}.")


def face_sharpness(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_face_embedding(person_crop, min_face_size=None, blur_threshold=None, verbose=False):
    """Detect the best face in a person crop (with quality filtering) and
    return its ArcFace embedding, or None if no usable face is visible."""
    min_face_size = MIN_FACE_SIZE_PX if min_face_size is None else min_face_size
    blur_threshold = FACE_BLUR_THRESHOLD if blur_threshold is None else blur_threshold

    with face_engine_lock:
        try:
            faces = face_app.get(person_crop)
        except Exception as e:
            if verbose:
                print(f"    InsightFace raised an error: {e}")
            return None

    if not faces:
        return None

    best_face, best_area, skip_reason = None, 0, None
    for face in faces:
        x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
        fw, fh = x2 - x1, y2 - y1
        if fw <= 0 or fh <= 0:
            continue
        if min(fw, fh) < min_face_size:
            skip_reason = f"face too small ({min(fw, fh)}px, need >= {min_face_size}px)"
            continue
        crop = person_crop[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            continue
        sharpness = face_sharpness(crop)
        if sharpness < blur_threshold:
            skip_reason = f"face too blurry (sharpness {sharpness:.0f}, need >= {blur_threshold})"
            continue
        area = fw * fh
        if area > best_area:
            best_area = area
            best_face = face

    if best_face is None:
        if verbose and skip_reason:
            print(f"    Face detected but rejected: {skip_reason}")
        return None

    return np.array(best_face.normed_embedding)


def cosine_distance(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1 - float(np.dot(a, b) / denom)


def identify_employee(person_crop):
    """Returns (name, embedding). Matches against the CLOSEST embedding in
    each employee's gallery (nearest-neighbor), not a single averaged
    template - much more forgiving of the live face being at a different
    angle than most of that employee's reference photos.

    Two checks must both pass for a name to be returned:
      1. The best-matching employee's closest embedding must clear
         EMPLOYEE_MATCH_THRESHOLD (as before).
      2. That best match must beat the SECOND-best-matching employee by at
         least MIN_MATCH_MARGIN. If two different employees are both about
         equally close, the read is too ambiguous to trust - it's better to
         return no name (and let another read, or the clothing assist,
         settle it) than to guess and risk swapping two people's names."""
    embedding = get_face_embedding(person_crop, verbose=DEBUG_FACE_MATCHING)
    if embedding is None:
        return None, embedding
    if not employee_gallery:
        if DEBUG_FACE_MATCHING:
            print("    No employee galleries built yet - nobody can be recognized "
                  f"(check that '{EMPLOYEE_PHOTOS_FOLDER}' has a subfolder with photos per person).")
        return None, embedding

    # Best distance PER EMPLOYEE (not just the single overall-closest
    # embedding) so the margin check below compares "how close is this
    # embedding to person A" vs "to person B", regardless of how many
    # reference photos either of them happens to have.
    best_per_employee = []
    for name, gallery in employee_gallery.items():
        best_for_name = min(cosine_distance(embedding, ref) for ref in gallery)
        best_per_employee.append((best_for_name, name))
    best_per_employee.sort(key=lambda pair: pair[0])

    best_dist, best_name = best_per_employee[0]
    if best_dist > EMPLOYEE_MATCH_THRESHOLD:
        if DEBUG_FACE_MATCHING:
            print(f"    Closest match '{best_name}' at distance {best_dist:.3f} - "
                  f"above threshold {EMPLOYEE_MATCH_THRESHOLD} (lower = more similar), no match.")
        return None, embedding

    if len(best_per_employee) > 1:
        second_dist, second_name = best_per_employee[1]
        if second_dist - best_dist < MIN_MATCH_MARGIN:
            print(f"    Ambiguous face match: '{best_name}' ({best_dist:.3f}) vs "
                  f"'{second_name}' ({second_dist:.3f}) - too close to call, skipping.")
            return None, embedding

    if DEBUG_FACE_MATCHING:
        print(f"    Matched '{best_name}' at distance {best_dist:.3f}.")
    return best_name, embedding


def _today_str():
    return datetime.date.today().isoformat()


def clothing_histogram(person_crop):
    """A rough HSV color-histogram 'fingerprint' of what someone's wearing,
    used only as a secondary re-identification hint (see CLOTHING_ASSIST
    settings). Skips the top ~30% of the crop (head/hair/skin, not clothing)
    and hashes color distribution in the rest. Returns None if the crop is
    too small to bother with."""
    h, w = person_crop.shape[:2]
    if h < 40 or w < 20:
        return None
    torso = person_crop[int(h * 0.30):h, :]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(CLOTHING_HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def clothing_distance(hist_a, hist_b):
    """1 - histogram correlation, so LOWER = more similar (same convention
    as cosine_distance for faces)."""
    corr = cv2.compareHist(hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL)
    return 1 - float(corr)


def learn_clothing(name, person_crop):
    """Update a face-confirmed employee's learned outfit for TODAY, as a
    running average (so it adapts a bit if lighting changes camera to
    camera). Automatically starts fresh whenever the calendar date changes,
    since - unlike a face - what someone's wearing isn't stable day to day."""
    if not CLOTHING_ASSIST_ENABLED:
        return
    hist = clothing_histogram(person_crop)
    if hist is None:
        return
    today = _today_str()
    with employee_clothing_lock:
        entry = employee_clothing.get(name)
        if entry is None or entry["date"] != today:
            employee_clothing[name] = {"date": today, "hist": hist, "samples": 1}
        else:
            n = min(entry["samples"], CLOTHING_LEARN_SAMPLES_PER_DAY)
            entry["hist"] = (entry["hist"] * n + hist) / (n + 1)
            entry["samples"] += 1


def match_clothing(person_crop):
    """Compare a live crop's clothing against every employee's outfit
    LEARNED SO FAR TODAY. Returns (name, distance) only for a clear,
    unambiguous match (see CLOTHING_MATCH_THRESHOLD/MARGIN) - otherwise
    (None, None). This never originates an identity on its own; it's only
    ever used to reinforce/extend one a face already established earlier
    the same day (see run_camera())."""
    if not CLOTHING_ASSIST_ENABLED:
        return None, None
    hist = clothing_histogram(person_crop)
    if hist is None:
        return None, None
    today = _today_str()
    with employee_clothing_lock:
        candidates = [(n, e["hist"]) for n, e in employee_clothing.items() if e["date"] == today]
    if not candidates:
        return None, None

    ranked = sorted((clothing_distance(hist, h), n) for n, h in candidates)
    best_dist, best_name = ranked[0]
    if best_dist > CLOTHING_MATCH_THRESHOLD:
        return None, None
    if len(ranked) > 1:
        second_dist, _ = ranked[1]
        if second_dist - best_dist < CLOTHING_MATCH_MARGIN:
            return None, None  # two employees dressed too similarly today - ambiguous
    return best_name, best_dist


def _employee_folder_signature(person_dir, valid_ext):
    entries = []
    for filename in sorted(os.listdir(person_dir)):
        if not filename.lower().endswith(valid_ext):
            continue
        path = os.path.join(person_dir, filename)
        try:
            stat = os.stat(path)
            entries.append((filename, stat.st_mtime, stat.st_size))
        except OSError:
            continue
    return entries


def build_employee_embeddings():
    """Build a GALLERY of embeddings per employee (one per usable reference
    photo, plus a horizontally-flipped version of each if
    EMPLOYEE_USE_FLIP_AUGMENTATION is on) from every photo in their subfolder
    of EMPLOYEE_PHOTOS_FOLDER. Cached to disk keyed by each employee's photo
    filenames/sizes/mtimes, so re-runs are instant unless their photos
    actually changed."""
    if not os.path.isdir(EMPLOYEE_PHOTOS_FOLDER):
        print(f"ERROR: '{EMPLOYEE_PHOTOS_FOLDER}' folder not found. Create it with one "
              f"subfolder per employee, each containing a few clear face photos.")
        return

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    cache = {}
    if os.path.exists(EMPLOYEE_CACHE_FILE):
        try:
            with open(EMPLOYEE_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
        except Exception as e:
            print(f"WARNING: could not read '{EMPLOYEE_CACHE_FILE}' ({e}) - rebuilding.")

    new_cache = {}
    for name in sorted(os.listdir(EMPLOYEE_PHOTOS_FOLDER)):
        person_dir = os.path.join(EMPLOYEE_PHOTOS_FOLDER, name)
        if not os.path.isdir(person_dir):
            continue

        signature = _employee_folder_signature(person_dir, valid_ext)
        cached = cache.get(name)
        if cached is not None and cached.get("signature") == signature and cached.get("gallery"):
            employee_gallery[name] = cached["gallery"]
            new_cache[name] = cached
            print(f"  '{name}': loaded from cache ({len(cached['gallery'])} embedding(s)).")
            continue

        gallery = []
        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith(valid_ext):
                continue
            photo_path = os.path.join(person_dir, filename)
            img = cv2.imread(photo_path)
            if img is None:
                print(f"  WARNING: could not read '{photo_path}' - skipping.")
                continue

            tall_side = max(img.shape[0], img.shape[1])
            if tall_side > EMPLOYEE_PHOTO_MAX_SIDE_PX:
                scale = EMPLOYEE_PHOTO_MAX_SIDE_PX / tall_side
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            variants = [img]
            if EMPLOYEE_USE_FLIP_AUGMENTATION:
                variants.append(cv2.flip(img, 1))

            found_any = False
            for variant in variants:
                emb = get_face_embedding(variant)
                if emb is None:
                    emb = get_face_embedding(
                        variant, verbose=True,
                        min_face_size=EMPLOYEE_MIN_FACE_SIZE_PX,
                        blur_threshold=EMPLOYEE_FACE_BLUR_THRESHOLD,
                    )
                if emb is None:
                    continue
                norm = np.linalg.norm(emb)
                if norm > 0:
                    gallery.append(emb / norm)
                    found_any = True

            if not found_any:
                print(f"  WARNING: no usable face found in '{photo_path}' - skipping.")

        if not gallery:
            print(f"  WARNING: '{name}' has no usable reference photos - they won't be recognized.")
            continue

        if len(gallery) > EMPLOYEE_GALLERY_MAX_PER_PERSON:
            gallery = gallery[:EMPLOYEE_GALLERY_MAX_PER_PERSON]

        employee_gallery[name] = gallery
        new_cache[name] = {"signature": signature, "gallery": gallery}
        print(f"  Built gallery for '{name}': {len(gallery)} embedding(s) "
              f"({'with' if EMPLOYEE_USE_FLIP_AUGMENTATION else 'without'} flip augmentation).")

    try:
        with open(EMPLOYEE_CACHE_FILE, "wb") as f:
            pickle.dump(new_cache, f)
    except Exception as e:
        print(f"WARNING: could not write '{EMPLOYEE_CACHE_FILE}' ({e}).")

    if not employee_gallery:
        print("WARNING: no employee face galleries were built - no one will be recognized.")
    else:
        total = sum(len(g) for g in employee_gallery.values())
        print(f"Built galleries for {len(employee_gallery)} employee(s), {total} embedding(s) total.")


# ------------------------------------------------------------- Camera I/O --

def open_camera(source):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def open_camera_with_timeout(source, label, timeout=CAMERA_OPEN_TIMEOUT_SECONDS):
    result = {"cap": None}

    def worker():
        result["cap"] = open_camera(source)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"[{label}] Connection attempt timed out after {timeout}s - will retry.")
        return None
    return result["cap"]


def reconnect_until_success(source, label, initial_delay=2, max_delay=30):
    delay = initial_delay
    while True:
        cap = open_camera_with_timeout(source, label)
        if cap is not None and cap.isOpened():
            return cap
        print(f"[{label}] Reconnect attempt failed - retrying in {delay}s...")
        time.sleep(delay)
        delay = min(delay * 2, max_delay)


class CameraStream:
    """Continuously reads frames from one camera on its own thread, so the
    detection loop is never waiting on network/decode latency - it just grabs
    whatever the newest frame is whenever it's ready to process one."""

    def __init__(self, source, label):
        self.source = source
        self.label = label
        self.frame = None
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.stopped = False
        self.cap = reconnect_until_success(source, label)
        threading.Thread(target=self._reader_loop, daemon=True).start()

    def _reader_loop(self):
        failures = 0
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self.cap = reconnect_until_success(self.source, self.label)
                failures = 0
                continue

            ok, frame = self.cap.read()
            stale = (time.time() - self.last_frame_time) > FRAME_STALE_SECONDS if self.last_frame_time else False

            if not ok or stale:
                failures += 1
                if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    print(f"[{self.label}] Lost connection - reconnecting...")
                    self.cap.release()
                    self.cap = reconnect_until_success(self.source, self.label)
                    failures = 0
                continue

            failures = 0
            with self.lock:
                self.frame = frame
                self.last_frame_time = time.time()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.cap is not None:
            self.cap.release()


# ------------------------------------------------------------------ Camera worker --

def draw_labeled_box(frame, x, y, w, h, label, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(frame, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_TEXT_COLOR, 1)


def run_camera(camera_config):
    label = camera_config["label"]
    source = camera_config["source"]
    role = camera_config.get("role", "interior")
    if role not in VALID_ROLES:
        print(f"[{label}] WARNING: unknown role '{role}' - treating as 'interior'.")
        role = "interior"
    ignore_zones = camera_config.get("ignore_zones", [])
    print(f"[{label}] Starting (role={role})...")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    stream = CameraStream(source, label)
    show_preview = SHOW_LIVE_PREVIEW
    if show_preview:
        cv2.namedWindow(label, cv2.WINDOW_NORMAL)

    # track_id -> {"name": str|None, "last_identified": float,
    #              "candidate_name": str|None, "candidate_count": int}
    local_tracks = {}
    last_check = 0
    # Boxes from the most recent detection pass, redrawn on preview frames in
    # between passes so the preview window stays smooth/responsive instead
    # of only updating once per CHECK_EVERY_SECONDS.
    last_boxes = []
    # Throttles how often the preview window actually redraws - see
    # PREVIEW_MAX_FPS. Independent of CHECK_EVERY_SECONDS/detection cadence.
    last_preview = 0
    preview_interval = 1.0 / PREVIEW_MAX_FPS if PREVIEW_MAX_FPS > 0 else 0

    while True:
        if stop_event.is_set():
            if show_preview:
                cv2.destroyWindow(label)
            print(f"[{label}] Stopped.")
            return

        frame = stream.get_frame()
        if frame is None:
            if show_preview:
                cv2.waitKey(1)  # keep the window's message pump alive while waiting
            time.sleep(0.1)
            continue

        now = time.time()
        run_detection = (now - last_check >= CHECK_EVERY_SECONDS)

        if run_detection:
            last_check = now
            try:
                results = yolo_model.track(
                    frame, classes=[PERSON_CLASS_ID], conf=PERSON_CONFIDENCE,
                    verbose=False, device=YOLO_DEVICE, persist=True, tracker=TRACKER_CONFIG,
                )
            except Exception as e:
                print(f"[{label}] Detection/tracking error: {e}")
                results = None

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                frame_boxes = []

                for (x1, y1, x2, y2), track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if DEBUG_PRINT_BOX_COORDS:
                        print(f"[{label}] DEBUG box coords: ({x1}, {y1}, {x2}, {y2})")

                    if in_ignore_zone(x1, y1, x2, y2, ignore_zones):
                        continue  # e.g. a passerby visible through a glass door - skip entirely

                    person_crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    if person_crop.size == 0:
                        continue

                    track = local_tracks.setdefault(track_id, {
                        "name": None, "last_identified": 0,
                        "candidate_name": None, "candidate_count": 0, "candidate_source": None,
                    })

                    # Only re-run face recognition periodically per track, and stop
                    # once locked on - nothing more to learn from re-checking an
                    # already-confirmed person every few seconds.
                    if track["name"] is None and now - track["last_identified"] >= REIDENTIFY_INTERVAL_SECONDS:
                        name, _ = identify_employee(person_crop)
                        id_source = "face"

                        if name is None and CLOTHING_ASSIST_ENABLED:
                            # Face was inconclusive (no usable face angle, or an
                            # ambiguous match) - fall back to today's learned outfit
                            # as a weaker secondary signal. See CLOTHING_ASSIST
                            # settings: this only fires once a face has already
                            # taught the script what this employee is wearing today.
                            name, _ = match_clothing(person_crop)
                            id_source = "clothing" if name is not None else None

                        track["last_identified"] = now

                        # Lock-on: a name only sticks once it's agreed with the
                        # previous read(s), AND those reads came from the same kind
                        # of evidence. Anything that breaks the streak (a different
                        # name, a source switch, or a frame with no usable match)
                        # resets the count - this is what keeps a single bad-angle
                        # frame, or an ambiguous read, from misidentifying someone.
                        if name is not None and name == track["candidate_name"] and id_source == track["candidate_source"]:
                            track["candidate_count"] += 1
                        else:
                            track["candidate_name"] = name
                            track["candidate_source"] = id_source
                            track["candidate_count"] = 1 if name is not None else 0

                        required_reads = (IDENTITY_LOCK_CONSECUTIVE_READS if track["candidate_source"] == "face"
                                           else CLOTHING_LOCK_CONSECUTIVE_READS)
                        if track["candidate_name"] is not None and track["candidate_count"] >= required_reads:
                            track["name"] = track["candidate_name"]

                    if track["name"] is not None:
                        handle_sighting(track["name"], label, role, now, frame_for_photo=frame)
                        learn_clothing(track["name"], person_crop)
                        frame_boxes.append((x1, y1, x2, y2, track["name"], BOX_COLOR_KNOWN))
                    elif show_preview:
                        frame_boxes.append((x1, y1, x2, y2, "Identifying...", BOX_COLOR_UNKNOWN))

                last_boxes = frame_boxes

                # Drop local tracks YOLO's tracker isn't reporting anymore (person
                # left this camera's view) - cheap bookkeeping only, doesn't affect
                # building-wide attendance state.
                seen_ids = set(track_ids.tolist())
                for tid in list(local_tracks.keys()):
                    if tid not in seen_ids:
                        del local_tracks[tid]
            else:
                last_boxes = []

        if show_preview:
            due_for_preview = (now - last_preview >= preview_interval)
            if due_for_preview:
                last_preview = now
                # frame is already a fresh copy from stream.get_frame() above -
                # draw straight onto it instead of copying it a second time.
                for (x1, y1, x2, y2, text, color) in last_boxes:
                    draw_labeled_box(frame, x1, y1, x2 - x1, y2 - y1, text, color)
                cv2.imshow(label, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyWindow(label)
                    print(f"[{label}] Preview window closed (camera keeps running in the background).")
                    show_preview = False
            else:
                # Still pump the window's event loop so it doesn't look frozen,
                # without paying for a full redraw/imshow this iteration.
                cv2.waitKey(1)
                if not run_detection:
                    time.sleep(0.01)
        elif not run_detection:
            time.sleep(0.03)


def camera_supervisor(camera_config):
    """Keeps a camera's worker running forever, restarting with backoff if it
    ever crashes (bad credentials, camera unreachable, unexpected error)."""
    label = camera_config["label"]
    delay = 5
    max_delay = 60
    while not stop_event.is_set():
        start_time = time.time()
        try:
            run_camera(camera_config)
            if stop_event.is_set():
                return
            print(f"[{label}] Worker stopped unexpectedly.")
        except Exception as e:
            print(f"[{label}] CRASHED: {e}")
            traceback.print_exc()

        if time.time() - start_time > 30:
            delay = 5  # it ran fine for a while - don't penalize a one-off blip
        print(f"[{label}] Restarting in {delay}s...")
        time.sleep(delay)
        delay = min(delay * 2, max_delay)


# ------------------------------------------------------------------------ main --

def main():
    if not CAMERAS:
        print("ERROR: no cameras configured - add at least one entry to CAMERAS.")
        return

    init_face_engine()
    print("Building employee face galleries (this runs once, may take a moment)...")
    build_employee_embeddings()

    threading.Thread(target=checkout_fallback_sweep_loop, daemon=True).start()

    camera_threads = []
    for camera_config in CAMERAS:
        t = threading.Thread(target=camera_supervisor, args=(camera_config,), daemon=True)
        t.start()
        camera_threads.append(t)

    print(f"Running {len(CAMERAS)} camera(s). Attendance log: {os.path.abspath(ATTENDANCE_CSV_PATH)}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping - closing camera connections and preview windows...")
        stop_event.set()
        for t in camera_threads:
            t.join(timeout=5)
        cv2.destroyAllWindows()  # safety net for any window that didn't self-close in time
        print("Stopped.")


if __name__ == "__main__":
    main()