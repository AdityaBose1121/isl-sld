"""Scan all datasets and print statistics."""
import os

DATA_ROOT = r"c:\Users\adity\OneDrive\Desktop\Projects\sign-language-detection2\data"

# === INCLUDE dataset ===
print("=" * 60)
print("INCLUDE Dataset")
print("=" * 60)
include_dir = os.path.join(DATA_ROOT, "include")
cats = sorted([c for c in os.listdir(include_dir)
               if os.path.isdir(os.path.join(include_dir, c))])
print(f"Categories: {len(cats)}")
total_signs = 0
total_videos = 0
for cat in cats:
    catpath = os.path.join(include_dir, cat)
    signs = sorted([s for s in os.listdir(catpath)
                    if os.path.isdir(os.path.join(catpath, s))])
    cat_vids = 0
    for s in signs:
        spath = os.path.join(catpath, s)
        vids = [v for v in os.listdir(spath)
                if v.lower().endswith(('.mov', '.mp4', '.avi', '.mkv'))]
        cat_vids += len(vids)
    total_signs += len(signs)
    total_videos += cat_vids
    print(f"  {cat}: {len(signs)} signs, {cat_vids} videos")
print(f"TOTAL: {total_signs} sign classes, {total_videos} videos")

# === ISL-CSLTR dataset ===
print()
print("=" * 60)
print("ISL-CSLTR Dataset")
print("=" * 60)
csltr_dir = os.path.join(DATA_ROOT, "isl-csltr")

# Word-level
word_dir = os.path.join(csltr_dir, "Frames_Word_Level")
word_classes = sorted([d for d in os.listdir(word_dir)
                       if os.path.isdir(os.path.join(word_dir, d))])
total_word_imgs = 0
for wc in word_classes:
    wcpath = os.path.join(word_dir, wc)
    imgs = [f for f in os.listdir(wcpath)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_word_imgs += len(imgs)
print(f"Word-level: {len(word_classes)} classes, {total_word_imgs} images")

# Sentence-level videos
sent_vid_dir = os.path.join(csltr_dir, "Videos_Sentence_Level")
sent_classes = sorted([d for d in os.listdir(sent_vid_dir)
                       if os.path.isdir(os.path.join(sent_vid_dir, d))])
total_sent_vids = 0
for sc in sent_classes:
    scpath = os.path.join(sent_vid_dir, sc)
    vids = [f for f in os.listdir(scpath)
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    total_sent_vids += len(vids)
print(f"Sentence-level: {len(sent_classes)} sentences, {total_sent_vids} videos")

# Sentence-level frames
sent_frm_dir = os.path.join(csltr_dir, "Frames_Sentence_Level")
sent_frm_classes = sorted([d for d in os.listdir(sent_frm_dir)
                           if os.path.isdir(os.path.join(sent_frm_dir, d))])
total_sent_frms = 0
for sf in sent_frm_classes:
    sfpath = os.path.join(sent_frm_dir, sf)
    # May have sub-dirs (signer variants)
    for root, dirs, files in os.walk(sfpath):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_sent_frms += 1
print(f"Sentence-level frames: {len(sent_frm_classes)} sentences, {total_sent_frms} frame images")

# === FER-2013 dataset ===
print()
print("=" * 60)
print("FER-2013 Dataset")
print("=" * 60)
fer_dir = os.path.join(DATA_ROOT, "fer2013")
for split in ["train", "test"]:
    split_dir = os.path.join(fer_dir, split)
    if not os.path.exists(split_dir):
        print(f"  {split}: NOT FOUND")
        continue
    emotions = sorted(os.listdir(split_dir))
    total_imgs = 0
    for em in emotions:
        empath = os.path.join(split_dir, em)
        if os.path.isdir(empath):
            imgs = len([f for f in os.listdir(empath)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_imgs += imgs
    print(f"  {split}: {total_imgs} images across {len(emotions)} emotions")
