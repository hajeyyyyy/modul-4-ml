#!/bin/bash
# =============================================================
# Lab 4.3: 3D Generation with NeRF or Gaussian Splatting
# =============================================================
# Objective: Rekonstruksi 3D scene dari foto menggunakan Nerfstudio
# 
# PLATFORM NOTES:
# - MacBook Air M3: Nerfstudio support MPS tapi lambat untuk NeRF.
#   Gunakan Gaussian Splatting (splatfacto) yang lebih ringan.
# - Google Colab T4: Lebih baik untuk ini, bisa handle NeRF penuh.
#
# INSTALL:
#   pip install nerfstudio
#   ns-install-cli   (install COLMAP dan dependencies)
#
# SETUP FOTO:
#   1. Ambil 30-50 foto objek kecil (cangkir, mainan, tanaman)
#   2. Pastikan objek terlit dengan baik dari semua sisi
#   3. Rotate sekitar objek 360° sambil tetap di level yang sama
#   4. Hindari motion blur, gunakan mode portrait/macro
# =============================================================

set -e  # Exit jika ada error

# ── CONFIG ──────────────────────────────────────────────────
PHOTOS_DIR="./photos"          # Folder foto-foto objek kamu
OUTPUT_DATA="./data/my_object"
OBJECT_NAME="my_object"

echo "=================================================="
echo "Lab 4.3: 3D Reconstruction with Nerfstudio"
echo "=================================================="

# ── STEP 1: Install Nerfstudio ───────────────────────────────
install_nerfstudio() {
    echo ""
    echo "[STEP 1] Installing Nerfstudio..."
    pip install nerfstudio --quiet
    
    # Install CLI tools (COLMAP untuk camera pose estimation)
    echo "Installing COLMAP dan dependencies..."
    ns-install-cli --mode install
    
    echo "✓ Nerfstudio terinstall"
}

# ── STEP 2: Siapkan Foto ─────────────────────────────────────
prepare_photos() {
    echo ""
    echo "[STEP 2] Menyiapkan folder foto..."
    
    if [ ! -d "$PHOTOS_DIR" ]; then
        mkdir -p "$PHOTOS_DIR"
        echo "⚠ Folder $PHOTOS_DIR dibuat."
        echo "  → Masukkan 30-50 foto objek ke folder ini lalu jalankan ulang."
        echo ""
        echo "TIPS ambil foto yang baik:"
        echo "  - Jarak konsisten dari objek (30-50cm)"
        echo "  - Pencahayaan stabil (tidak ada bayangan keras)"
        echo "  - Rotate mengelilingi objek setiap 10-15 derajat"
        echo "  - Sertakan beberapa foto dari atas dan bawah"
        exit 0
    fi
    
    PHOTO_COUNT=$(ls "$PHOTOS_DIR"/*.{jpg,jpeg,JPG,JPEG,png,PNG} 2>/dev/null | wc -l)
    echo "✓ Ditemukan $PHOTO_COUNT foto di $PHOTOS_DIR"
    
    if [ "$PHOTO_COUNT" -lt 20 ]; then
        echo "⚠ Disarankan minimal 20-30 foto untuk rekonstruksi yang baik"
    fi
}

# ── STEP 3: Process Images (Estimasi Camera Poses) ──────────
process_images() {
    echo ""
    echo "[STEP 3] Processing images dengan COLMAP..."
    echo "  (Estimasi camera poses dari semua foto)"
    
    ns-process-data images \
        --data "$PHOTOS_DIR" \
        --output-dir "$OUTPUT_DATA" \
        --num-downscales 1     # Downscale untuk speed
    
    echo "✓ Camera poses diestimasi, data tersimpan di $OUTPUT_DATA"
}

# ── STEP 4A: Train Gaussian Splatting (Rekomendasi untuk M3) ──
train_gaussian_splatting() {
    echo ""
    echo "[STEP 4A] Training Gaussian Splatting (splatfacto)..."
    echo "  ⚡ Lebih cepat dari NeRF, real-time rendering"
    echo "  ⏱ Estimasi waktu: 5-15 menit"
    
    ns-train splatfacto \
        --data "$OUTPUT_DATA" \
        --viewer.quit-on-train-completion True \
        --max-num-iterations 15000   # Kurangi untuk speed
    
    echo "✓ Gaussian Splatting training selesai"
    echo "  → Buka http://localhost:7007 di browser untuk melihat 3D viewer"
}

# ── STEP 4B: Train NeRF (Instant-NGP, lebih cepat dari vanilla NeRF) ─
train_nerf() {
    echo ""
    echo "[STEP 4B] Training NeRF dengan Instant-NGP (nerfacto)..."
    echo "  ⏱ Estimasi waktu: 10-30 menit"
    
    ns-train nerfacto \
        --data "$OUTPUT_DATA" \
        --viewer.quit-on-train-completion True
    
    echo "✓ NeRF training selesai"
}

# ── STEP 5: Render Video Flythrough ─────────────────────────
render_video() {
    echo ""
    echo "[STEP 5] Rendering video flythrough..."
    
    # Cari config file terbaru
    LATEST_CONFIG=$(find outputs/ -name "config.yml" -newer "$OUTPUT_DATA" | sort | tail -1)
    
    if [ -z "$LATEST_CONFIG" ]; then
        echo "⚠ Config file tidak ditemukan. Jalankan training dulu."
        return 1
    fi
    
    echo "  Config: $LATEST_CONFIG"
    
    # Render dengan camera path spiral
    ns-render interpolate \
        --load-config "$LATEST_CONFIG" \
        --output-path "output_video.mp4" \
        --render-output-names rgb \
        --interpolation-steps 120
    
    echo "✓ Video disimpan ke output_video.mp4"
}

# ── STEP 6: Eksport 3D Model ────────────────────────────────
export_model() {
    echo ""
    echo "[STEP 6] Eksport ke format mesh (untuk Blender, dll)..."
    
    LATEST_CONFIG=$(find outputs/ -name "config.yml" | sort | tail -1)
    
    ns-export poisson \
        --load-config "$LATEST_CONFIG" \
        --output-dir "./exported_mesh" \
        --target-num-faces 200000
    
    echo "✓ Mesh diekspor ke ./exported_mesh/"
    echo "  → Bisa dibuka di Blender, MeshLab, atau viewer online"
}

# ── ALTERNATIF: Hunyuan 3D Studio (Tanpa Kode) ─────────────
alternative_hunyuan() {
    echo ""
    echo "=================================================="
    echo "ALTERNATIF: Hunyuan 3D Studio (Tidak perlu kode!)"
    echo "=================================================="
    echo ""
    echo "1. Kunjungi: https://3d.hunyuan.tencent.com/"
    echo "2. Upload foto objek ATAU masukkan text prompt"
    echo "3. Download 3D model dalam format GLB/FBX"
    echo "4. Buka di Blender untuk melihat hasilnya"
    echo ""
    echo "Cocok untuk:"
    echo "  - text-to-3D: 'a wooden chair with curved legs'"
    echo "  - image-to-3D: upload foto produk/objek apapun"
    echo "  - 3D editing: modifikasi model yang sudah ada"
}

# ── ACTIVITY ANSWERS ──────────────────────────────────────────
print_activity_answers() {
    echo ""
    echo "=================================================="
    echo "ACTIVITY ANSWERS — Lab 4.3"
    echo "=================================================="
    
    cat << 'EOF'

Q1: Perbedaan Representasi Explicit vs Implicit 3D?

EXPLICIT (Mesh, Point Cloud, Voxels, Gaussian Splatting):
  - Definisi langsung posisi dan warna titik/poligon di 3D space
  - Mudah diedit, rendering cepat (rasterization), mudah diintegrasikan ke engine
  - Gaussian Splatting: real-time 60-100 FPS
  - Kekurangan: membutuhkan mesh yang kompleks untuk detail halus

IMPLICIT (NeRF, SDF):
  - 3D scene dikodekan sebagai fungsi kontinu f(x,y,z) → (color, density)
  - Representasi continuous, arbitrary resolution
  - Kelebihan: sangat halus, tidak ada polygon artifacts
  - Kekurangan: rendering lambat (butuh raycasting yang mahal), sulit diedit

---
Q2: Kenapa Gaussian Splatting Lebih Cepat dari NeRF?

NeRF: Untuk setiap pixel, cast ray dan sample 64-192 titik sepanjang ray,
      setiap titik di-forward melalui MLP. O(pixels × samples × MLP_cost)

Gaussian Splatting: Rasterisasi Gaussian ke image plane secara langsung,
                   bisa diakselerasi dengan GPU rasterizer yang sangat optimal.
                   O(N_gaussians × projection_cost) — jauh lebih efisien.

---
Q3: Aplikasi NeRF di Robotik?

1. Scene Reconstruction: Robot bisa "berjalan" di sekitar environment baru
   dan membangun model 3D untuk navigasi tanpa peta manual.
2. Manipulation Planning: NeRF scene bisa digunakan untuk merencanakan
   gerakan arm robot untuk mengambil objek dari sudut pandang yang berbeda.
3. Sim-to-Real: Render gambar dari NeRF scene untuk training vision policy,
   membantu robot belajar di simulasi sebelum deploy di dunia nyata.

---
Q4: Tantangan Scaling Text-to-3D ke Scene Kompleks?

1. Consistency: Diffusion model dilatih per-view, sulit menjaga 3D consistency
   dari banyak sudut (masalah Janus face — objek terlihat benar dari 1 sudut
   tapi salah dari sudut lain)
2. Compute: SDS optimization butuh ratusan forward pass diffusion model
3. Ambiguity: Teks seperti "a room" sangat ambigu untuk layout 3D
4. Scale: Full scene punya depth complexity yang jauh melebihi objek tunggal

EOF
}

# ── MAIN MENU ────────────────────────────────────────────────
echo ""
echo "Pilih operasi:"
echo "  1) Full pipeline (install → process → train Gaussian Splatting → render)"
echo "  2) Hanya train NeRF"
echo "  3) Hanya render video (sudah punya trained model)"
echo "  4) Tampilkan alternatif Hunyuan 3D Studio"
echo "  5) Tampilkan Activity Answers"
echo ""
read -p "Masukkan pilihan [1-5]: " choice

case $choice in
    1)
        install_nerfstudio
        prepare_photos
        process_images
        train_gaussian_splatting
        render_video
        print_activity_answers
        ;;
    2)
        prepare_photos
        process_images
        train_nerf
        render_video
        ;;
    3)
        render_video
        export_model
        ;;
    4)
        alternative_hunyuan
        ;;
    5)
        print_activity_answers
        ;;
    *)
        echo "Pilihan tidak valid"
        alternative_hunyuan
        print_activity_answers
        ;;
esac

echo ""
echo "✓ Lab 4.3 selesai!"