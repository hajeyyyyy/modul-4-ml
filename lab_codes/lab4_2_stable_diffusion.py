"""
Lab 4.2: Image Generation with Stable Diffusion
=================================================
Objective: Generate images from text prompts menggunakan diffusers library.
Eksperimen: guidance scale, negative prompts, inference steps, seeds.

PLATFORM NOTES:
- Google Colab T4 (CUDA): gunakan float16, IMG_SIZE=512 oke
- MacBook Air M3 (MPS): gunakan float32, IMG_SIZE=256 untuk kecepatan
  - MPS support Stable Diffusion tapi lebih lambat dari CUDA
  - Untuk M3: jalankan di Colab atau gunakan Core ML / mlx-lm sebagai alternatif
- CPU only: IMG_SIZE=128 saja, butuh beberapa menit per gambar

INSTALL:
    pip install diffusers transformers accelerate pillow torch
"""

import torch
import os
from PIL import Image
import time

# ============================================================
# 0. PLATFORM & CONFIG
# ============================================================
def get_device():
    if torch.cuda.is_available():
        dtype = torch.float16
        print("✓ CUDA GPU — mode Colab T4 (float16)")
    elif torch.backends.mps.is_available():
        dtype = torch.float32  # MPS lebih stabil dengan float32
        print("✓ Apple MPS — mode MacBook M-series (float32)")
    else:
        dtype = torch.float32
        print("⚠ CPU mode — akan lambat")
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    return device, dtype

DEVICE, DTYPE = get_device()

# Config
MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMG_SIZE = 256 if DEVICE in ["mps", "cpu"] else 512   # Kecil untuk M3
OUTPUT_DIR = "./sd_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD PIPELINE
# ============================================================
def load_pipeline():
    from diffusers import StableDiffusionPipeline

    print(f"\n[Loading] {MODEL_ID}...")
    print("(Download ~4GB saat pertama kali, cached setelahnya)")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,       # Disable untuk inference lebih cepat
        requires_safety_checker=False,
    )

    if DEVICE == "mps":
        pipe = pipe.to("mps")
        # Aktifkan attention slicing untuk hemat memory di M3
        pipe.enable_attention_slicing()
        print("✓ Attention slicing diaktifkan (hemat memory M3)")
    elif DEVICE == "cuda":
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    else:
        # CPU: gunakan lighter model jika terlalu lambat
        print("⚠ CPU mode aktif — pertimbangkan Colab untuk kecepatan optimal")
        pipe = pipe.to("cpu")

    return pipe

# ============================================================
# 2. GENERATE IMAGES — BERBAGAI PROMPT
# ============================================================
def generate_sample_images(pipe):
    print("\n" + "="*60)
    print("PART 1: Sample Image Generation")
    print("="*60)

    prompts = [
        {
            "prompt": "A serene mountain landscape at sunset, photorealistic, 4k, golden hour",
            "negative": "blurry, low quality, distorted, overexposed",
            "filename": "mountain_landscape.png",
        },
        {
            "prompt": "A futuristic city with flying cars, neon lights, cyberpunk style, rain",
            "negative": "blurry, low resolution, cartoon, anime",
            "filename": "cyberpunk_city.png",
        },
        {
            "prompt": "A cute cat wearing a tiny astronaut suit floating in space, digital art",
            "negative": "ugly, deformed, low quality",
            "filename": "space_cat.png",
        },
    ]

    for i, item in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating: {item['prompt'][:60]}...")
        t0 = time.time()

        image = pipe(
            prompt=item["prompt"],
            negative_prompt=item["negative"],
            height=IMG_SIZE,
            width=IMG_SIZE,
            num_inference_steps=20,    # 20 cukup untuk preview
            guidance_scale=7.5,
            generator=torch.manual_seed(42),
        ).images[0]

        path = os.path.join(OUTPUT_DIR, item["filename"])
        image.save(path)
        elapsed = time.time() - t0
        print(f"✓ Saved: {path} ({elapsed:.1f}s)")

# ============================================================
# 3. EKSPERIMEN GUIDANCE SCALE
# ============================================================
def experiment_guidance_scale(pipe):
    print("\n" + "="*60)
    print("PART 2: Eksperimen Guidance Scale")
    print("="*60)
    print("Guidance scale mengontrol seberapa kuat prompt mempengaruhi output.")
    print("Rendah = lebih kreatif/random, Tinggi = lebih patuh pada prompt\n")

    prompt = "A dragon breathing fire in a medieval fantasy landscape, detailed, dramatic lighting"
    negative = "blurry, cartoon, low quality"
    guidance_scales = [2.0, 5.0, 7.5, 12.0]

    results = []
    for gs in guidance_scales:
        print(f"  Generating guidance_scale={gs}...")
        t0 = time.time()
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            height=IMG_SIZE,
            width=IMG_SIZE,
            num_inference_steps=20,
            guidance_scale=gs,
            generator=torch.manual_seed(42),
        ).images[0]

        filename = f"dragon_gs_{gs}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        image.save(path)
        elapsed = time.time() - t0
        results.append((gs, path, elapsed))
        print(f"  ✓ gs={gs}: {filename} ({elapsed:.1f}s)")

    print("\nObservasi:")
    print("  gs=2.0  → Gambar bebas, mungkin tidak persis dragon")
    print("  gs=7.5  → Balance antara kreativitas dan prompt adherence")
    print("  gs=12.0 → Sangat literal ke prompt, tapi bisa artefak")
    return results

# ============================================================
# 4. EKSPERIMEN INFERENCE STEPS
# ============================================================
def experiment_inference_steps(pipe):
    print("\n" + "="*60)
    print("PART 3: Eksperimen Jumlah Inference Steps")
    print("="*60)

    prompt = "A beautiful watercolor painting of a Japanese cherry blossom garden"
    steps_list = [5, 10, 20, 40]

    for steps in steps_list:
        print(f"  Generating steps={steps}...")
        t0 = time.time()
        image = pipe(
            prompt=prompt,
            height=IMG_SIZE,
            width=IMG_SIZE,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=torch.manual_seed(42),
        ).images[0]

        path = os.path.join(OUTPUT_DIR, f"steps_{steps}.png")
        image.save(path)
        elapsed = time.time() - t0
        print(f"  ✓ steps={steps}: {elapsed:.1f}s")

    print("\nObservasi:")
    print("  5 steps  → Sangat cepat tapi noisy/blurry")
    print("  10 steps → Mulai terlihat struktur gambar")
    print("  20 steps → Quality yang baik, sweet spot")
    print("  40 steps → Sedikit lebih detail, waktu 2× lebih lama")

# ============================================================
# 5. EKSPERIMEN NEGATIVE PROMPTS
# ============================================================
def experiment_negative_prompts(pipe):
    print("\n" + "="*60)
    print("PART 4: Eksperimen Negative Prompts")
    print("="*60)

    prompt = "A portrait of a person smiling"

    # Tanpa negative prompt
    print("  Tanpa negative prompt...")
    image_no_neg = pipe(
        prompt=prompt,
        height=IMG_SIZE, width=IMG_SIZE,
        num_inference_steps=20, guidance_scale=7.5,
        generator=torch.manual_seed(42),
    ).images[0]
    image_no_neg.save(os.path.join(OUTPUT_DIR, "portrait_no_negative.png"))

    # Dengan negative prompt
    negative_prompt = "blurry, low quality, distorted face, extra fingers, deformed, ugly, watermark, text"
    print("  Dengan negative prompt...")
    image_with_neg = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=IMG_SIZE, width=IMG_SIZE,
        num_inference_steps=20, guidance_scale=7.5,
        generator=torch.manual_seed(42),
    ).images[0]
    image_with_neg.save(os.path.join(OUTPUT_DIR, "portrait_with_negative.png"))

    print("✓ Bandingkan: portrait_no_negative.png vs portrait_with_negative.png")
    print(f"\nNegative prompt yang digunakan:\n  '{negative_prompt}'")

# ============================================================
# JAWABAN ACTIVITY QUESTIONS
# ============================================================
def print_activity_answers():
    print("\n" + "="*60)
    print("ACTIVITY ANSWERS — Lab 4.2")
    print("="*60)

    answers = {
        "Q1 - Efek guidance_scale dari 2 ke 15": """
        - guidance_scale=2 : Gambar kreatif dan bebas, tidak terlalu patuh pada prompt.
          Bisa menghasilkan interpretasi unik yang tidak terduga.
        - guidance_scale=7.5: Sweet spot — balance kreativitas dan prompt adherence.
          Default yang baik untuk sebagian besar use case.
        - guidance_scale=15 : Sangat literal mengikuti prompt, warna lebih saturasi,
          tapi cenderung menghasilkan artefak atau tampak "terlalu tajam".
        KESIMPULAN: guidance_scale=7-8 menghasilkan hasil paling seimbang.
        """,

        "Q2 - Pengaruh num_inference_steps": """
        - 5 steps  : Sangat cepat (~2s di CUDA), kualitas sangat buruk (noise/blob)
        - 10 steps : Mulai terlihat struktur dasar gambar, acceptable untuk sketch
        - 20 steps : Sweet spot — kualitas baik, 2-5s di CUDA, sweet spot
        - 40 steps : Sedikit lebih detail, waktu 2× lebih lama, diminishing returns
        - 50+ steps: Hampir tidak ada perbedaan kualitas vs 40 steps
        MINIMUM ACCEPTABLE: 15-20 steps untuk review, 30+ untuk produksi.
        """,

        "Q3 - Peran negative prompts dan 3 contoh": """
        Negative prompts menginstruksikan model untuk MENGHINDARI konsep-konsep
        tertentu saat generasi. Secara matematis, ini menggunakan classifier-free
        guidance dengan menggeser arah denoising menjauh dari kondisi negatif.
        
        3 contoh negative prompt yang meningkatkan kualitas:
        1. "blurry, low quality, low resolution, pixelated" 
           → Memaksa output lebih tajam dan detail
        2. "extra fingers, deformed hands, mutated body parts, ugly"
           → Kritis untuk portrait/karakter manusia
        3. "text, watermark, signature, logo, username"
           → Mencegah teks random muncul di gambar
        """,

        "Q4 - Classifier-Free Guidance (CFG)": """
        CFG bekerja dengan dua forward pass:
        1. Conditional: epsilon_cond = model(x_t, t, text_embedding)
        2. Unconditional: epsilon_uncond = model(x_t, t, empty_embedding)
        
        Final score: epsilon_final = epsilon_uncond + guidance_scale × (epsilon_cond - epsilon_uncond)
        
        Intuisi: arahkan prediksi noise ke arah yang "lebih sesuai kondisi" dari 
        yang tidak-kondisional, dengan kekuatan yang dikontrol oleh guidance_scale.
        Ini secara efektif mengamplifikasi pengaruh text conditioning.
        """,
    }

    for q, a in answers.items():
        print(f"\n{q}:{a}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LAB 4.2: Image Generation with Stable Diffusion")
    print(f"Device: {DEVICE} | Image Size: {IMG_SIZE}×{IMG_SIZE}")
    print("=" * 60)

    # Load pipeline
    pipe = load_pipeline()

    # Jalankan eksperimen
    print("\n[INFO] Semua output akan disimpan di:", OUTPUT_DIR)

    generate_sample_images(pipe)
    experiment_guidance_scale(pipe)
    experiment_inference_steps(pipe)
    experiment_negative_prompts(pipe)
    print_activity_answers()

    print(f"\n✓ Lab 4.2 selesai!")
    print(f"Semua gambar tersimpan di: {OUTPUT_DIR}/")
    print("\nFile yang dihasilkan:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            print(f"  - {f}")