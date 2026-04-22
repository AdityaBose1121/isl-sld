"""Quick test: verify INCLUDE dataset scanning and name extraction works."""
import re
import os

INCLUDE_DIR = r"c:\Users\adity\OneDrive\Desktop\Projects\sign-language-detection2\data\include"

categories = sorted([
    d for d in os.listdir(INCLUDE_DIR)
    if os.path.isdir(os.path.join(INCLUDE_DIR, d))
])

print(f"Categories found: {len(categories)}")
all_signs = {}  # sign_label -> (category, original_dir, video_count)

for category in categories:
    cat_path = os.path.join(INCLUDE_DIR, category)
    sign_dirs = sorted([
        d for d in os.listdir(cat_path)
        if os.path.isdir(os.path.join(cat_path, d))
    ])

    for sign_dir in sign_dirs:
        # Parse: "1. loud" -> "loud", "83. big large" -> "big_large"
        clean_name = re.sub(r'^\d+\.\s*', '', sign_dir)
        clean_name = clean_name.strip().lower().replace(' ', '_')
        sign_path = os.path.join(cat_path, sign_dir)
        vids = len([f for f in os.listdir(sign_path)
                    if f.lower().endswith(('.mov', '.mp4', '.avi'))])

        if clean_name in all_signs:
            print(f"  DUPLICATE: '{clean_name}' in {category} AND {all_signs[clean_name][0]}")
        all_signs[clean_name] = (category, sign_dir, vids)

print(f"\nTotal unique sign labels: {len(all_signs)}")
print(f"\nSample sign name mappings:")
for i, (label, (cat, orig, vids)) in enumerate(sorted(all_signs.items())):
    if i < 20 or i >= len(all_signs) - 5:
        print(f"  {orig:30s} -> {label:20s} ({cat}, {vids} videos)")
    elif i == 20:
        print(f"  ... ({len(all_signs) - 25} more) ...")

# Check for any signs with 0 videos
empty = [(l, c) for l, (c, _, v) in all_signs.items() if v == 0]
if empty:
    print(f"\nWARNING: {len(empty)} signs have 0 videos: {empty}")
else:
    print("\nAll signs have at least 1 video. Ready for extraction!")
