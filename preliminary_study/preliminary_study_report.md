# Preliminary Study Report — Module 4: Generative AI
**Undergraduate Machine Learning Practicum**

---

## Q1. Autoregressive Language Models

### Apa itu Autoregressive Generation?

Autoregressive generation adalah metode generasi teks di mana model menghasilkan satu token pada satu waktu, dengan setiap token dikondisikan pada semua token sebelumnya. Model ini memfaktorisasi probabilitas joint dari sebuah sekuens menggunakan chain rule of probability:

$$p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})$$

Pada saat inferensi, model:
1. Menerima konteks awal (prompt)
2. Memprediksi distribusi probabilitas atas seluruh vocabulary untuk token berikutnya
3. Sampling satu token dari distribusi tersebut
4. Menambahkan token ke konteks
5. Mengulangi proses hingga kriteria berhenti terpenuhi (token [EOS] atau panjang maksimum)

### GPT (Decoder-Only) vs BERT (Encoder-Only)

| Aspek | GPT (Decoder-Only) | BERT (Encoder-Only) |
|---|---|---|
| Attention | Causal (unidirectional) — hanya bisa lihat token sebelumnya | Bidirectional — bisa lihat semua token |
| Training objective | Next-token prediction | Masked Language Modeling (MLM) |
| Cocok untuk | Text generation, chatbot, code completion | Classification, NER, QA extraction |
| Representasi | Autoregressive, token-by-token | Seluruh sekuens sekaligus |

GPT lebih cocok untuk text generation karena arsitektur causal mask-nya secara alami cocok dengan proses generasi kiri-ke-kanan. BERT tidak bisa digunakan langsung untuk generasi teks karena ia membutuhkan seluruh konteks kanan-kiri untuk menghasilkan representasi yang baik.

### Temperature dalam Text Generation

Temperature τ mengontrol "keacakan" distribusi output dengan menskala logits sebelum softmax:

$$p(x_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

- **τ → 0 (misal 0.1):** Output hampir deterministik, selalu memilih token dengan probabilitas tertinggi. Teks lebih koheren tapi repetitif.
- **τ = 1.0:** Distribusi standar softmax, keseimbangan antara kreativitas dan koherensi.
- **τ = 2.0:** Distribusi lebih flat/seragam, sampling lebih random. Output lebih kreatif/beragam tapi bisa inkoheren.

### Top-k dan Top-p (Nucleus) Sampling

**Top-k sampling:** Hanya sampling dari k token dengan probabilitas tertinggi. Misalnya k=50 berarti hanya 50 kandidat token terbaik yang dipertimbangkan. Cocok digunakan saat kita ingin membatasi variasi tapi tetap memberi ruang kreativitas.

**Top-p (nucleus) sampling:** Sampling dari set token terkecil yang jumlah probabilitas kumulatifnya melebihi p (misalnya p=0.9). Jumlah kandidat bervariasi tergantung distribusi — jika satu token sangat dominan, hanya itu yang dipilih; jika distribusi merata, banyak token masuk kandidat. Lebih adaptif dibanding top-k.

Contoh penggunaan:
- **Top-k (k=10):** Saat menulis kode — hanya ingin token yang benar-benar relevan secara sintaksis.
- **Top-p (p=0.9):** Saat menulis cerita kreatif — ingin variasi alami sesuai konteks.

---

## Q2. Diffusion Models — Conceptual Foundations

### Forward Diffusion Process

Proses forward diffusion secara bertahap menambahkan Gaussian noise ke data bersih x₀ selama T langkah menggunakan jadwal noise β₁, β₂, ..., βT:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

Setelah T langkah, x_T ≈ N(0, I) — pure Gaussian noise. Dengan reparameterization trick, kita bisa langsung sampling x_t dari x₀:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

di mana $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$.

### Reverse Denoising Process

Model belajar membalikkan proses noise: dari x_T (pure noise), secara iteratif memprediksi dan menghapus noise untuk mendapatkan x_{T-1}, x_{T-2}, ..., x₀. Model neural network ε_θ(x_t, t) dilatih untuk memprediksi noise ε yang ditambahkan pada langkah t, dengan loss:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### Peran UNet Architecture

UNet efektif untuk diffusion karena:
1. **Encoder-decoder symmetry:** Downsampling mengekstrak fitur high-level, upsampling merekonstruksi detail spasial.
2. **Skip connections:** Menggabungkan fitur encoder ke decoder, mempertahankan detail lokal yang penting untuk rekonstruksi gambar.
3. **Multi-scale processing:** Menangkap struktur gambar pada berbagai resolusi secara bersamaan.
4. **Time conditioning:** Timestep t di-embed dan diinjeksikan ke setiap layer agar model tahu seberapa banyak noise yang ada.

### Cross-Attention untuk Text-to-Image

Untuk mengkondisikan generasi gambar pada teks, UNet menggunakan cross-attention:
- Text encoder (CLIP) mengkonversi prompt menjadi sequence embeddings c = {c₁, ..., c_L}
- Di setiap layer UNet, feature gambar (query Q) attend ke token teks (key K, value V):

$$\text{CrossAttention}(Q_\text{img}, K_\text{text}, V_\text{text}) = \text{softmax}\left(\frac{Q_\text{img} K_\text{text}^\top}{\sqrt{d}}\right) V_\text{text}$$

Ini memungkinkan setiap region gambar "mencari" bagian teks yang paling relevan saat denoising.

### Latent Diffusion Models (Stable Diffusion)

Diffusion langsung di pixel space untuk gambar 1024×1024 sangat mahal secara komputasi. LDM menyelesaikan ini dengan:
1. **VAE Encoder:** Kompres gambar x (H×W×3) ke latent z (H/8 × W/8 × 4) — kompresi 48×.
2. **Diffusion di latent space:** Semua proses noise/denoise di z-space (jauh lebih kecil).
3. **VAE Decoder:** Rekonstruksi gambar dari latent yang sudah di-denoise.

Manfaat komputasi: training dan inferensi 48× lebih cepat dan hemat memori dibanding pixel-space diffusion.

### Diffusion Transformers (DiT)

DiT mengganti UNet dengan pure Vision Transformer:
- Gambar/latent dibagi menjadi patch → sequence of tokens
- Standard Transformer blocks (multi-head self-attention + MLP)
- Adaptive Layer Norm (adaLN) untuk mengkondisikan pada timestep t dan text conditioning
- Scaling lebih baik dengan ukuran model; state-of-the-art di ImageNet generation
- Digunakan di OpenAI Sora (video), FLUX (gambar)

---

## Q3. GANs vs. Diffusion

### Core Idea GANs

GAN (Goodfellow et al., 2014) terdiri dari dua jaringan yang dilatih secara adversarial:
- **Generator G:** Memetakan random noise z ~ N(0, I) ke sampel fake G(z)
- **Discriminator D:** Mengklasifikasikan input sebagai real (dari dataset) atau fake (dari G)

Minimax objective:

$$\min_G \max_D \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

### Tantangan Training GAN

1. **Mode collapse:** Generator hanya mempelajari beberapa mode dari distribusi data, menghasilkan output yang kurang beragam.
2. **Training instability:** Keseimbangan Nash antara G dan D sulit dicapai; loss bisa berosilasi atau diverge.
3. **Vanishing gradients:** Jika D terlalu kuat, gradient ke G menjadi sangat kecil sehingga G tidak belajar efektif.
4. **Hyperparameter sensitivity:** Sangat sensitif terhadap learning rate, arsitektur, dan scheduling.

### Keunggulan Diffusion atas GAN

1. **Training stabil:** Loss function sederhana (MSE), tidak ada adversarial game yang tidak stabil.
2. **Diversity lebih baik:** Diffusion memodelkan full distribusi data; tidak rentan terhadap mode collapse.
3. **Conditioning mudah:** Cross-attention secara natural mengintegrasikan teks atau kondisi lain.
4. **Controllability:** Inpainting, outpainting, img2img, ControlNet semua mudah diimplementasi.
5. **Skalabilitas:** DiT menunjukkan scaling yang sangat baik dengan ukuran model.

### Domain GAN Masih Kompetitif

- **Real-time inference:** GAN menghasilkan gambar dalam satu forward pass; diffusion butuh 20–50 langkah.
- **Audio synthesis:** HiFi-GAN, MelGAN sangat efisien untuk konversi spectrogram ke waveform.
- **Style transfer & super-resolution:** Pix2Pix, CycleGAN untuk image translation paired/unpaired.
- **Artistic control:** StyleGAN latent space manipulation untuk manipulasi atribut wajah.

---

## Q4. RLHF and Alignment

### Mengapa SFT pada Web Text Tidak Cukup?

Pre-training pada web text mengoptimasi "apa yang manusia tulis di internet," bukan "apa yang helpful, harmless, dan honest." Perilaku yang muncul dari pure next-token prediction:
- Mengikuti distribusi konten internet yang penuh bias, toksisitas, dan misinformasi
- Meniru berbagai gaya penulisan tanpa filter kualitas
- Tidak memahami intensi pengguna; cenderung meneruskan pola teks daripada menjawab secara helpful
- Dapat dengan percaya diri menghasilkan informasi palsu (hallucination)

### RLHF Pipeline (3 Tahap)

**Tahap 1 — Supervised Fine-Tuning (SFT):**
- Kumpulkan demonstrasi berkualitas tinggi: human labeler menulis contoh respons ideal untuk berbagai prompt
- Fine-tune LLM pre-trained pada demonstrasi ini menggunakan standard next-token prediction loss
- Hasil: model yang bisa mengikuti instruksi, tapi belum optimal

**Tahap 2 — Training Reward Model:**
- Untuk banyak prompt, generate beberapa respons dari SFT model
- Human labeler meranking respons mana yang lebih baik (A vs B comparison)
- Train reward model r_φ(x, y) menggunakan Bradley-Terry pairwise ranking loss untuk memprediksi preferensi manusia
- Output reward model: scalar reward score untuk pasangan (prompt, response)

**Tahap 3 — RL Fine-Tuning (PPO):**
- Treat LLM sebagai RL policy: action space = vocabulary, state = prompt + partial response
- Reward function dari reward model tahap 2
- Optimasi menggunakan Proximal Policy Optimization (PPO)
- Tambahkan KL-divergence penalty terhadap SFT model untuk mencegah reward hacking dan distributional shift

### Direct Preference Optimization (DPO)

DPO (Rafailov et al., 2023) menyederhanakan RLHF dengan mengeliminasi reward model terpisah dan RL training:

Key insight: optimal policy di bawah RLHF objective bisa diekspresikan dalam closed form sebagai fungsi dari reference policy. DPO langsung mengoptimasi policy menggunakan preference pairs (x, y_w, y_l):

$$\mathcal{L}_\text{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

Keunggulan: tidak perlu reward model, tidak perlu RL, lebih stabil, lebih mudah diimplementasi. Secara empiris matches atau melebihi RLHF pada banyak benchmark.

### RLAIF (Reinforcement Learning from AI Feedback)

RLAIF menggunakan AI kuat (misal GPT-4) sebagai pengganti human labeler untuk memberikan preference label. Proses:
1. Generate pasangan respons dari SFT model
2. Gunakan "AI annotator" (GPT-4) untuk memilih respons yang lebih baik
3. Train reward model dari AI preferences ini
4. Jalankan PPO seperti RLHF biasa

**Pros:** Sangat scalable, murah, bisa generate preference data dalam jumlah besar, konsisten.
**Cons:** Mewarisi bias dari AI annotator, bisa menciptakan echo chamber jika AI annotator dan policy model dari keluarga yang sama, kurang ground truth kebenaran manusia.

### Safety Concern: Reward Hacking

**Masalah:** Model belajar mengeksploitasi kelemahan reward model, bukan memenuhi intensi aslinya. Contoh: menghasilkan respons verbose yang panjang mendapat reward tinggi karena terlihat "komprehensif" meski isinya tidak akurat.

**Mitigasi:**
- Ensemble reward models dari beberapa checkpoints untuk robustness
- KL penalty yang kuat terhadap SFT reference model
- Constitutional AI: model melakukan self-critique berdasarkan prinsip tertulis
- Evaluasi berkala dengan human rater pada held-out test set
- Red-teaming adversarial untuk menemukan exploit sebelum deployment

---

## Q5. Parameter-Efficient Fine-Tuning (PEFT)

### Mengapa Full Fine-Tuning 70B Model Tidak Praktis?

Fine-tuning model Llama-3-70B penuh membutuhkan:
- **Memory weights (float32):** 70B × 4 bytes = 280 GB
- **Gradients:** +280 GB
- **Optimizer states (AdamW):** +560 GB (momentum + variance)
- **Total:** ~1.1 TB VRAM — membutuhkan ~14 GPU A100 80GB

Ini di luar kemampuan sebagian besar researcher dan mahasiswa. PEFT memungkinkan adaptasi dengan sumber daya yang jauh lebih sedikit.

### LoRA: Low-Rank Adaptation

LoRA membekukan weight pre-trained W dan menambahkan matriks dekomposisi low-rank yang trainable:

$$W' = W + \Delta W = W + BA$$

di mana B ∈ ℝ^{d×r}, A ∈ ℝ^{r×d}, dan r ≪ d.

### Analisis Parameter LoRA

Untuk weight matrix W berukuran d × d:
- **Full fine-tuning:** d² parameter trainable
- **LoRA rank r:** 2 × d × r parameter trainable
- **Rasio:** (2dr) / d² = 2r/d

Contoh dengan d = 4096, r = 8:
- Full fine-tuning: 4096² = **16,777,216** parameter
- LoRA: 2 × 4096 × 8 = **65,536** parameter
- Penghematan: 99.6%

### QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) menggabungkan LoRA dengan 4-bit quantization:
- **Base model dalam 4-bit:** Menggunakan NormalFloat (NF4) quantization. 70B model: 280 GB → ~35 GB
- **LoRA adapters dalam 16-bit (bfloat16):** B dan A tetap di presisi tinggi untuk training stabil
- **Double quantization:** Juga mengkuantisasi quantization constants untuk hemat memori lebih lanjut
- **Hasil:** Fine-tune 70B model di satu GPU A6000 48GB

### Adapter vs Prefix Tuning vs LoRA

**Adapters:**
- Menyisipkan MLP bottleneck kecil di antara layer Transformer
- Hanya adapter yang dilatih, base model dibekukan
- ~1-3% parameter tambahan
- Kekurangan: menambah inference latency karena layer tambahan di forward pass

**Prefix Tuning:**
- Menambahkan "virtual tokens" trainable di awal input ke setiap layer
- Sangat sedikit parameter (~0.01-0.1%)
- Kekurangan: mengurangi effective context length karena sebagian context digunakan untuk prefix

**LoRA:**
- Lebih efisien dari adapter (bisa di-merge, tidak ada inference overhead)
- Lebih banyak parameter dari prefix tuning tapi lebih ekspresif
- Pilihan terbaik untuk sebagian besar use case

---

## Q6. Modern Frontiers in Generative AI

### 3D Generation

**NeRF (Neural Radiance Fields):**
NeRF merepresentasikan scene 3D sebagai fungsi volumetrik kontinu:

$$F_\theta: (x, y, z, \theta, \phi) \rightarrow (r, g, b, \sigma)$$

di mana (x, y, z) adalah posisi 3D, (θ, φ) adalah arah pandang, (r, g, b) adalah warna, dan σ adalah volume density. Training menggunakan 50–100 gambar dengan pose kamera yang diketahui, mengoptimasi MLP kecil via volumetric ray marching dan photometric loss.

**Gaussian Splatting:**
Merepresentasikan scene sebagai set 3D Gaussians (posisi, covariance, warna, opacity). Rendering via fast rasterization. Keunggulan: real-time rendering (60–100 FPS), training dalam hitungan menit, representasi eksplisit yang mudah diedit.

**Hunyuan 3D Studio (Tencent):** Platform terintegrasi untuk text-to-3D, image-to-3D, dan 3D editing menggunakan diffusion-based 3D generation.

**Trellis 3D (Microsoft Research):** Model generatif 3D yang mengkombinasikan sparse voxel representation dengan Transformer untuk multi-view generation dan texture synthesis.

Pentingnya untuk VR/AR (konten imersif), robotics (simulasi environment), dan game development (asset generation).

### Video Generation

Text-to-video menambahkan tantangan temporal ke image diffusion:
- **Temporal coherence:** Frame harus konsisten satu sama lain
- **Motion coherence:** Objek harus bergerak secara fisik masuk akal
- **Long-range dependencies:** Awal dan akhir video harus konsisten

**OpenAI Sora:** Menggunakan spacetime latent patches + DiT, bisa menghasilkan video hingga 60 detik di 1080p. Bersifat proprietary.
**Runway Gen-2:** Video hingga 16 detik, tersedia via API.
**Pika:** Video pendek (3–5 detik), lebih accessible.

### Code Agents dan Agentic Behavior

Code-generating agent mengkombinasikan LLM dengan sandboxed execution environment:
1. User memberi prompt
2. LLM men-generate kode Python
3. Kode dieksekusi di sandbox aman
4. Output (gambar, error, dataframe) dikirim kembali ke LLM
5. LLM memperbaiki atau melanjutkan berdasarkan output

Contoh: GPT-4 Code Interpreter, Claude Code (Anthropic), Open Interpreter.

**Safety mechanisms:** Sandbox isolation (tidak bisa akses filesystem atau network host), resource limits (CPU, memory, timeout), whitelist library yang aman, audit logging.

### Synthetic Data Generation

Generative models membuat labeled training data untuk skenario di mana data real langka:
- Medical imaging: X-ray sintetis untuk penyakit langka
- Low-resource languages: Generasi teks untuk bahasa minority menggunakan LLM
- Safety-critical: Simulasi edge case untuk autonomous driving

**Risks:**
- **Model collapse:** Training pada data sintetis secara iteratif menurunkan performa ("model autophagy")
- **Bias amplification:** Bias dari generator terinherit dan teramplifikasi dalam data sintetis

**Mitigasi:** Selalu campur data sintetis dengan data real, validasi di held-out real data, gunakan diversity-promoting sampling.

### New Tools

**Leanstral:** LLM khusus untuk matematika formal dan theorem proving. Dilatih pada kode Lean (proof assistant language), dapat menggenerate dan memverifikasi bukti matematis. Use case: formalisasi otomatis research paper, edukasi formal methods.

**Google Stitch:** AI-powered UI/UX design tool yang menggunakan generative model untuk membuat wireframe, UI components, dan desain antarmuka dari prompt teks. Bagian dari ekosistem Google AI Studio, memungkinkan rapid prototyping desain aplikasi tanpa coding manual.

---

*Referensi utama: Ho et al. (DDPM, 2020), Rombach et al. (LDM/Stable Diffusion, 2022), Ouyang et al. (InstructGPT, 2022), Rafailov et al. (DPO, 2023), Hu et al. (LoRA, 2021), Mildenhall et al. (NeRF, 2020), Kerbl et al. (3DGS, 2023)*