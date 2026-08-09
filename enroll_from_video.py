"""
Video Enrollment Helper - Build employee face-photo folders from videos
------------------------------------------------------------------------------
Drop short enrollment videos into per-employee subfolders - have each person
slowly turn their head left, right, and back to center over ~10-20 seconds,
in decent light - like this:

    enroll_videos/
        Jane Doe/
            clip1.mp4
            clip2.mp4
        John Smith/
            turn_left_right.mov

Then just run this script with no arguments. For every video found under
every employee subfolder, it will:

  1. Sample frames across the video.
  2. Run the SAME face-detection + quality filtering darvoza.py already uses
     (reject anything too small/blurry - a bad frame here would just make
     recognition worse, not better).
  3. Pick a spread-out, non-redundant subset of frames - different angles
     and lighting, rather than N near-identical frames of them looking
     straight ahead - by greedily choosing frames whose face EMBEDDINGS are
     most different from each other (farthest-point sampling), not just
     frames spaced evenly in time. Two frames a second apart while someone's
     mid-turn are more useful to the gallery than two frames where they
     happened to both be looking dead ahead.
  4. Save the selected frames as JPEGs into employees/<name>/, exactly where
     darvoza.py already expects employee reference photos - <name> is taken
     straight from the subfolder name.

If an employee has multiple videos, all of them are pooled together and the
diverse-frame selection runs once across the whole pool (so --max-photos is
a per-employee cap, not per-video).

By default, any employee subfolder whose employees/<name>/ folder already
has photos in it is SKIPPED entirely (videos aren't even opened) - this is
what makes it safe to just re-run the script after dropping new employees'
videos into enroll_videos/, without it re-processing everyone who was
already enrolled. Pass --force to re-process (and add more photos for)
already-enrolled employees too.

After running this, either delete employee_face_cache.pkl or just restart
darvoza.py - it will pick up the new photos through its existing gallery-
building pipeline. No changes to darvoza.py are needed.

Usage:
    python enroll_from_video.py

    # Optional tuning:
    python enroll_from_video.py --videos-dir enroll_videos \
        --sample-every 0.4 --max-photos 15

    # Process just one employee's subfolder:
    python enroll_from_video.py --only "Jane Doe"

    # Re-process/add more photos even for already-enrolled employees:
    python enroll_from_video.py --force

Run this from the same folder as darvoza.py (it imports it directly to
reuse the exact same face engine, quality thresholds, and distance metric -
so what gets saved here matches exactly what darvoza.py will later match
against).
"""

import argparse
import os

import cv2

import darvoza

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")


def sample_frames(video_path, sample_every_seconds):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps * sample_every_seconds)))

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def extract_candidates(frames):
    """Run face detection + quality filtering on each sampled frame, reusing
    darvoza.py's own thresholds (with the same strict-then-loose two-pass
    fallback it uses when building employee galleries). Returns a list of
    (frame, embedding) for every frame that had a usable face."""
    candidates = []
    total = len(frames)
    for i, frame in enumerate(frames, start=1):
        emb = darvoza.get_face_embedding(frame)
        if emb is None:
            emb = darvoza.get_face_embedding(
                frame,
                min_face_size=darvoza.EMPLOYEE_MIN_FACE_SIZE_PX,
                blur_threshold=darvoza.EMPLOYEE_FACE_BLUR_THRESHOLD,
            )
        if emb is not None:
            candidates.append((frame, emb))
        print(f"    Processed frame {i}/{total} ({len(candidates)} usable so far)...", end="\r")
    print()  # move past the progress line
    return candidates


def farthest_point_sample(candidates, max_photos):
    """Greedily pick the subset of candidates whose face embeddings are most
    spread out from each other in embedding space, so the saved photos cover
    different angles/lighting instead of near-duplicates of the same look."""
    if len(candidates) <= max_photos:
        return candidates

    chosen = [0]
    remaining = list(range(1, len(candidates)))
    while len(chosen) < max_photos and remaining:
        best_idx, best_min_dist = None, -1
        for r in remaining:
            min_dist = min(
                darvoza.cosine_distance(candidates[r][1], candidates[c][1])
                for c in chosen
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = r
        chosen.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in chosen]


def already_enrolled(name):
    """True if employees/<name>/ already exists and has at least one photo
    in it - i.e. this person has already been enrolled before (via this
    script, the regular photo workflow, or both)."""
    person_dir = os.path.join(darvoza.EMPLOYEE_PHOTOS_FOLDER, name)
    if not os.path.isdir(person_dir):
        return False
    return any(
        f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        for f in os.listdir(person_dir)
    )


def find_employee_videos(videos_dir, only=None):
    """Walk videos_dir for per-employee subfolders and return
    {employee_name: [video_path, ...]}, skipping anything that isn't a
    directory and any subfolder with no video files in it."""
    if not os.path.isdir(videos_dir):
        raise SystemExit(
            f"'{videos_dir}' folder not found. Create it with one subfolder per employee "
            f"(e.g. {videos_dir}/Jane Doe/clip.mp4) and put their enrollment video(s) inside."
        )

    employee_videos = {}
    for name in sorted(os.listdir(videos_dir)):
        person_dir = os.path.join(videos_dir, name)
        if not os.path.isdir(person_dir):
            continue
        if only and name != only:
            continue
        videos = sorted(
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        )
        if videos:
            employee_videos[name] = videos
    return employee_videos


def enroll_employee(name, video_paths, sample_every, max_photos):
    """Process every video for one employee, pool the usable-face frames
    together, and save the most diverse subset into employees/<name>/."""
    all_candidates = []
    for video_path in video_paths:
        print(f"  Sampling '{video_path}' every {sample_every}s...")
        frames = sample_frames(video_path, sample_every)
        print(f"    {len(frames)} frame(s) sampled.")
        if not frames:
            print(f"    WARNING: no frames could be read - check the path/codec. Skipping.")
            continue

        candidates = extract_candidates(frames)
        print(f"    {len(candidates)} frame(s) had a usable face.")
        all_candidates.extend(candidates)

    if not all_candidates:
        print(
            f"  No usable faces found for '{name}'. Try: better/more even lighting, framing "
            "the face larger in the shot, or a slower head turn so fewer frames are motion-blurred."
        )
        return 0

    print(f"  Selecting up to {max_photos} diverse-angle frame(s) from {len(all_candidates)} candidate(s)...")
    selected = farthest_point_sample(all_candidates, max_photos)

    person_dir = os.path.join(darvoza.EMPLOYEE_PHOTOS_FOLDER, name)
    os.makedirs(person_dir, exist_ok=True)
    existing = len([
        f for f in os.listdir(person_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])

    for i, (frame, _) in enumerate(selected):
        out_path = os.path.join(person_dir, f"video_enroll_{existing + i + 1:03d}.jpg")
        cv2.imwrite(out_path, frame)

    print(f"  Saved {len(selected)} photo(s) to '{person_dir}'.")
    return len(selected)


def main():
    parser = argparse.ArgumentParser(
        description="Automatically build employee reference-photo folders from enrollment "
                     "videos organized in per-employee subfolders."
    )
    parser.add_argument("--videos-dir", default="enroll_videos",
                         help="Folder containing one subfolder per employee, each holding that "
                              "employee's enrollment video(s) (default: enroll_videos).")
    parser.add_argument("--only", default=None,
                         help="Only process this one employee subfolder name, instead of all of them.")
    parser.add_argument("--sample-every", type=float, default=0.4,
                         help="Seconds between sampled frames (default: 0.4).")
    parser.add_argument("--max-photos", type=int, default=15,
                         help="Max reference photos to keep per employee (default: 15).")
    parser.add_argument("--force", action="store_true",
                         help="Also process employees who already have enrolled photos in "
                              "employees/<name>/ (by default, those are skipped - see module docstring).")
    args = parser.parse_args()

    employee_videos = find_employee_videos(args.videos_dir, only=args.only)
    if not employee_videos:
        target = f"'{args.only}' inside " if args.only else ""
        raise SystemExit(
            f"No video files found for {target}'{args.videos_dir}'. Expected layout: "
            f"{args.videos_dir}/<Employee Name>/<video files>."
        )

    if not args.force:
        already_done = [name for name in employee_videos if already_enrolled(name)]
        for name in already_done:
            print(f"Skipping '{name}' - already has enrolled photos in "
                  f"'{darvoza.EMPLOYEE_PHOTOS_FOLDER}/{name}' (use --force to process anyway).")
            del employee_videos[name]
        if not employee_videos:
            print("\nNothing to do - every employee subfolder here is already enrolled. "
                  "Use --force to re-process them anyway.")
            return

    print("Loading face engine...")
    darvoza.init_face_engine()

    total_saved = 0
    for name, video_paths in employee_videos.items():
        print(f"\n=== {name} ({len(video_paths)} video(s)) ===")
        total_saved += enroll_employee(name, video_paths, args.sample_every, args.max_photos)

    print(f"\nDone. Enrolled {len(employee_videos)} employee(s), {total_saved} photo(s) total.")
    print("Delete employee_face_cache.pkl (or just restart darvoza.py) to pick these up.")


if __name__ == "__main__":
    main()
