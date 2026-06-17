#!/bin/bash
# Multi-task v3 eğitim launcher + Drive sync döngüsü
set -e

export LD_LIBRARY_PATH=/usr/lib64-nvidia
export DATASET_ROOT=/content/FakeAVCeleb_v1.2
export WORK_DIR=/content/work
export DRIVE_DIR=/content/drive/MyDrive/Grup11_Deepfake_Results

mkdir -p "$WORK_DIR" "$DRIVE_DIR"

# Drive sync watcher (her 5 dk'da bir)
(
    while sleep 300; do
        rsync -a --no-perms --no-owner --no-group \
            --exclude='*.tmp' \
            "$WORK_DIR"/ "$DRIVE_DIR"/ 2>>"$WORK_DIR/sync.log" || echo "sync err" >> "$WORK_DIR/sync.log"
        echo "$(date +%H:%M:%S) synced" >> "$WORK_DIR/sync.log"
    done
) &
SYNC_PID=$!
echo "Drive sync watcher: PID=$SYNC_PID"

# Eğitim — train + eval + ablation
cd /content
python3 -u deepfake_v3.py --stage train --epochs 12 --batch 32 --workers 6 2>&1 \
    | tee "$WORK_DIR/train.log"

# Eğitim bitince final sync ve değerlendirme
rsync -a --no-perms --no-owner --no-group "$WORK_DIR"/ "$DRIVE_DIR"/ 2>>"$WORK_DIR/sync.log"

echo "=== EVAL STAGE ==="
python3 -u deepfake_v3.py --stage eval --batch 32 --workers 6 2>&1 \
    | tee -a "$WORK_DIR/train.log"
rsync -a --no-perms --no-owner --no-group "$WORK_DIR"/ "$DRIVE_DIR"/ 2>>"$WORK_DIR/sync.log"

echo "=== ABLATION STAGE ==="
python3 -u deepfake_v3.py --stage ablation --batch 32 --workers 6 2>&1 \
    | tee -a "$WORK_DIR/train.log"
rsync -a --no-perms --no-owner --no-group "$WORK_DIR"/ "$DRIVE_DIR"/ 2>>"$WORK_DIR/sync.log"

kill $SYNC_PID 2>/dev/null || true
echo "ALL STAGES DONE"
