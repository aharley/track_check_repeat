#!/bin/bash

FPS=24
THING="any_eval"
OUTPUT="${THING}.mp4"
rm -vf ${OUTPUT}
ffmpeg -framerate $FPS -pattern_type glob -i "tcr_vis/${THING}*.png" -s:v 768x832 -c:v libx264 -profile:v high -crf 1 -pix_fmt yuv420p $OUTPUT
open ${OUTPUT}
