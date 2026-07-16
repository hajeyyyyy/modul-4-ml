# Module 4 — Generative AI: LLMs, Images, Audio, Video, and 3D
**Undergraduate Machine Learning Practicum**  
Last updated: April 7, 2026

---

## Struktur Folder

```
PADL_M4_Project/
├── README.md                         ← File ini
├── preliminary_study/
│   └── preliminary_study_report.md   ← Jawaban Q1–Q6
├── lab_codes/
│   ├── lab4_1_lora_finetune.py       ← LoRA Fine-Tuning (Unsloth)
│   ├── lab4_2_stable_diffusion.py    ← Image Generation (Stable Diffusion)
│   ├── lab4_3_nerf_gaussian.sh       ← 3D Generation (Nerfstudio)
│   ├── lab4_4_toy_diffusion.py       ← Tiny DDPM from Scratch
│   └── lab4_5_mobile_llm.py          ← Mobile LLM Deployment
├── analysis_assignment/
│   └── analysis_report.md            ← Jawaban A1–A6
└── references/
    └── references.md                 ← Daftar paper dan resource
```

## Setup Environment (MacBook Air M3)

```bash
# Buat virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies umum
pip install torch torchvision torchaudio    # PyTorch untuk Apple Silicon (MPS)
pip install transformers datasets accelerate
pip install diffusers
pip install scikit-learn matplotlib numpy
pip install langchain langchain-community faiss-cpu sentence-transformers

# Untuk Lab 4.1 (LoRA)
pip install unsloth trl

# Untuk Lab 4.4 (Diffusion)
# Tidak perlu install tambahan, cukup paket di atas
```

> **Catatan M3:** Ganti `device = "cuda"` menjadi `device = "mps"` di semua script
> untuk memanfaatkan GPU Apple Silicon. Atau biarkan `"cpu"` untuk kompatibilitas penuh.
> Lab berat (4.1, 4.2) lebih baik dijalankan di Google Colab T4.

## Cara Jalankan

| Lab | Command |
|-----|---------|
| Lab 4.1 | `python lab_codes/lab4_1_lora_finetune.py` |
| Lab 4.2 | `python lab_codes/lab4_2_stable_diffusion.py` |
| Lab 4.3 | `bash lab_codes/lab4_3_nerf_gaussian.sh` |
| Lab 4.4 | `python lab_codes/lab4_4_toy_diffusion.py` |
| Lab 4.5 | Lihat komentar di `lab4_5_mobile_llm.py` |