#!/bin/bash

# Pindah ke direktori proyek
cd /Users/ridho/.gemini/antigravity/scratch/stunting-anthropometric-leakage

# Cek apakah ada perubahan file yang belum di-commit
if [ -n "$(git status --porcelain)" ]; then
    # Jika ada perubahan, jalankan sinkronisasi
    git add .
    git commit -m "Auto sync: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Supaya tidak ada masalah dengan SSH key saat berjalan di background
    export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"
    
    git push origin main
fi
