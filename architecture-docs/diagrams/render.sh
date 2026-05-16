#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
mkdir -p out

# Prefer magick (ImageMagick 7+), fall back to convert (ImageMagick 6)
if command -v magick &>/dev/null; then
    CONVERT="magick"
elif command -v convert &>/dev/null; then
    CONVERT="convert"
else
    CONVERT=""
fi

TARGET_W="${TARGET_W:-800}"
TARGET_H="${TARGET_H:-418}"

for src in *.d2; do
    name="${src%.d2}"
    echo ">>> $name"
    d2 --pad 20 "$src" "out/${name}.svg"
    d2 --pad 20 "$src" "out/${name}.png"
done

if [[ -n "$CONVERT" ]]; then
    echo "--- resizing to ${TARGET_W}x${TARGET_H} ---"
    for src in out/*.png; do
        name="${src%.png}"
        mv "$src" "${name}.raw.png"
        $CONVERT "${name}.raw.png" -resize "${TARGET_W}x${TARGET_H}" \
            -background white -gravity center \
            -extent "${TARGET_W}x${TARGET_H}" "${name}.png"
        rm -f "${name}.raw.png"
    done
fi

echo "Done. Outputs in ./out/"
