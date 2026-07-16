# Analysis Assignment — Module 4: Generative AI
**Undergraduate Machine Learning Practicum**

---

## A1. Autoregressive vs Diffusion untuk Image Generation

### Perbandingan Konseptual: Bagaimana Setiap Model Memodelkan p(x)?

**Autoregressive Models** memfaktorisasi distribusi gambar sebagai produk kondisional:

$$p(x) = \prod_{i=1}^{N} p(x_i \mid x_1, \ldots, x_{i-1})$$

Gambar diperlakukan sebagai sequence pixel (raster-scan order) atau patch token. Model memprediksi pixel/token berikutnya berdasarkan semua yang sudah ada. DALL-E 1 menggunakan VQVAE untuk tokenize gambar, kemudian GPT-style Transformer memprediksi sequence token gambar.

**Diffusion Models** mendekati distribusi gambar melalui proses denoising:

$$p(x_0) = \int p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t) \, dx_{1:T}$$

Model belajar membalikkan proses Markov yang secara bertahap mengaburkan data. Distribusi target didekati melalui Evidence Lower Bound (ELBO) yang dipermudah menjadi denoising score matching.

### Mengapa Diffusion Menggantikan Autoregressive untuk Image Synthesis?

1. **Global coherence:** Autoregressive model menghasilkan pixel secara lokal, rentan terhadap inkonsistensi global (bagian kiri dan kanan gambar mungkin tidak koheren). Diffusion beroperasi di seluruh gambar sekaligus.
2. **Speed training:** Diffusion bisa direct sample timestep sembarang di training; AR harus proses sequence secara sequential.
3. **Better conditioning:** Cross-attention di diffusion secara natural mengintegrasikan teks; AR butuh teknik khusus.
4. **Scalability:** DiT menunjukkan scaling yang lebih baik dari Transformer AR untuk gambar.
5. **Diversity:** AR model cenderung mode-seeking di distribusi gambar; diffusion lebih baik menangkap multi-modal distribusi.

### Keunggulan yang Masih Dimiliki Autoregressive Models

- **Exact likelihood:** AR memberikan exact log-likelihood; diffusion hanya ELBO.
- **Streaming/progressive generation:** Token-by-token generation natural untuk display progress.
- **Fine-grained controllability:** Bisa di-condition pada arbitrary previous tokens.
- **Code dan text:** Masih dominan untuk non-image modalities.

### Tabel Perbandingan

| Kriteria | Autoregressive (DALL-E 1) | Diffusion (SDXL, FLUX) |
|---|---|---|
| Training stability | Stabil (teacher-forcing) | Stabil (MSE loss) |
| Sample quality | Baik tapi kalah vs diffusion | State-of-the-art |
| Inference speed | Lambat (O(N) sequential steps) | Bisa lebih cepat (10-50 steps vs N tokens) |
| Controllability (inpainting) | Sulit — butuh resampling khusus | Mudah — mask + noise region |
| Conditional generation | Natural melalui context | Natural melalui cross-attention |
| Global coherence | Bisa inkonsisten antar region | Tinggi — denoise seluruh gambar |
| Likelihood estimation | Exact | Lower bound (ELBO) |
| Text conditioning | Moderate | Sangat baik (CLIP cross-attention) |

---

## A2. Analisis Matematis LoRA

### Bagaimana LoRA Memodifikasi Parameter?

LoRA membekukan weight pre-trained W ∈ ℝ^{d×d} dan menambahkan dekomposisi low-rank:

$$W' = W + \Delta W = W + BA$$

di mana B ∈ ℝ^{d×r} dan A ∈ ℝ^{r×d}, dengan r ≪ d.

Hanya B dan A yang dilatih. W tetap frozen dan tidak membutuhkan gradient.

Saat forward pass: `output = xW' = xW + xBA = xW + (xB)A`

Ini bisa dihitung secara efisien: pertama `h = xW` (pretrained path), lalu tambahkan `xBA` (adapter path), dengan scaling `(α/r)`.

### Derivasi Jumlah Trainable Parameters

Untuk satu weight matrix W ∈ ℝ^{d×d}:
- Full fine-tuning: **d²** parameter
- LoRA rank r: **d×r (B) + r×d (A) = 2dr** parameter

Rasio parameter: `2dr / d² = 2r/d`

Contoh konkret (d = 4096, r = 16):
- Full fine-tuning: 4096² = **16,777,216** parameter
- LoRA: 2 × 4096 × 16 = **131,072** parameter
- **Penghematan: 99.2%**

### Kalkulasi Total LoRA Parameters untuk Transformer Besar

**Konfigurasi:**
- d_model = 4096
- 32 layers
- LoRA diterapkan ke W_q dan W_v di setiap layer
- LoRA rank r = 16

Kalkulasi:
```
Parameter per matrix (W_q atau W_v) = 2 × d_model × r
                                     = 2 × 4096 × 16
                                     = 131,072

Matrices per layer = 2 (W_q + W_v)
Total per layer = 2 × 131,072 = 262,144

Total LoRA params = 32 layers × 262,144 = 8,388,608 ≈ 8.4M params
```

### Perbandingan dengan Full Model Parameters

Untuk Transformer standar dengan d_model = 4096, d_ff = 4 × 4096 = 16384:

Per layer (Transformer block):
- Attention (W_q, W_k, W_v, W_o): 4 × 4096² = 67,108,864
- MLP (W_1, W_2): 4096 × 16384 × 2 = 134,217,728
- Layer norms: 2 × 4096 × 2 ≈ negligible
- Total per layer: ≈ 201M params

Total model (32 layers + embedding ~4096×50000):
- Attention + MLP: 32 × 201M ≈ **6.4B params**
- Embedding: ~200M params
- **Total: ~6.6B params**

LoRA menggunakan 8.4M dari 6.6B: **0.127% dari total model parameters!**

### Trade-off LoRA Rank

| Rank r | Parameters | Use Case |
|--------|-----------|----------|
| r = 1 | Minimum — hanya 1 dimension | Sangat sedikit, hampir tidak efektif |
| r = 4 | ~2M params | Ringan, task sederhana (style, tone) |
| r = 16 | ~8M params | Sweet spot umum untuk task-specific |
| r = 64 | ~33M params | Task kompleks, banyak domain knowledge |
| r = 256 | ~134M params | Mendekati full fine-tuning, overfitting risk |

**Kapan r=4 lebih baik:** Style transfer, tone alignment, simple format following, few-shot adaptation. Dataset kecil (<1000 contoh).

**Kapan r=64 lebih baik:** Pembelajaran domain knowledge yang kuat (medical, legal, coding), dataset besar, task yang butuh reasoning kompleks.

---

## A3. Implikasi Etis dan Sosial Generative AI

### (a) Deepfakes dan Misinformasi

**Masalah:** Model generatif dapat menciptakan gambar, video, dan audio yang tidak dapat dibedakan dari konten asli. Implikasinya mencakup:
- **Jurnalisme:** Foto/video palsu dapat menyebarkan berita bohong dengan kecepatan viral, mempersulit fact-checking real-time.
- **Politik:** Video deepfake kandidat politik mengucapkan kata-kata yang tidak pernah diucapkan dapat mempengaruhi hasil pemilu.
- **Kepercayaan publik:** "Liar's dividend" — bahkan media asli dapat dibantah sebagai deepfake, mengikis kepercayaan terhadap semua bukti visual.
- **Pelecehan personal:** Deepfake pornografi non-konsensual (NCII) menarget individu, terutama perempuan.

**Mitigasi teknis:**
- **C2PA (Coalition for Content Provenance and Authenticity):** Standar metadata kriptografi yang menyematkan informasi asal konten. Adobe, Microsoft, Google mendukung standar ini.
- **Watermarking tidak terlihat:** SynthID (Google DeepMind) menyematkan watermark persisten dalam gambar/video yang tahan terhadap kompresi.
- **Adversarial perturbations:** Teknik seperti PhotoGuard menambahkan perturbasi tidak terlihat ke gambar asli sehingga jika dicuri untuk training, model menghasilkan output yang buruk.
- **Deepfake detection models:** Model seperti FakeCatcher (Intel) menganalisis subtle physiological signals.

### (b) Copyright dan Intellectual Property

**Masalah:** Diffusion model dilatih pada miliaran gambar internet yang sebagian besar dilindungi hak cipta. Ketika model menghasilkan gambar bergaya seniman tertentu, siapa yang berhak atas output tersebut?

Isu kunci:
- Artists seperti Greg Rutkowski mendapati namanya digunakan sebagai prompt untuk meniru gayanya tanpa kompensasi.
- Getty Images menggugat Stability AI karena training pada koleksinya tanpa lisensi.
- Model terkadang "meregurgitasi" konten training persis — apakah ini pelanggaran?

**Frameworks hukum yang ada dan dibutuhkan:**
- **Fair use doctrine (US):** Training AI bisa berargumen "transformatif" — belum ada preseden definitif.
- **Opt-out mechanisms:** HaveIBeenTrained.com, Spawning AI memungkinkan seniman meminta penghapusan karya dari dataset training.
- **Attribution metadata:** Setiap gambar yang di-generate bisa menyertakan model weights mana yang berkontribusi.
- **Licensing frameworks:** Model bisnis baru di mana seniman mendapatkan royalti ketika gaya mereka digunakan (Nightcafe, beberapa platform bereksperimen ini).

**Mitigasi teknis:**
- Training hanya pada data berlisensi (Adobe Stock + CC-licensed images → Adobe Firefly).
- Filtering dataset untuk menghapus konten yang jelas dilindungi.
- Consent tools: seniman bisa "opt-in" dan mendapat kompensasi.

### (c) Bias dalam Data dan Ketidakadilan

**Masalah:** Model dilatih pada web data yang mencerminkan dan memperparah bias sosial:
- **Gender bias:** Prompt "CEO" sering menghasilkan gambar pria; "nurse" sering perempuan.
- **Racial bias:** "Beautiful person" lebih sering menghasilkan kulit putih di beberapa model.
- **Language bias:** LLM jauh lebih baik dalam bahasa Inggris, menginferiorisasi pengguna bahasa lain.
- **Code bias:** Model coding lebih akurat untuk Python/JavaScript vs bahasa seperti Swahili.

**Manifestasi konkret:**
- Resume screening tools yang mendiskriminasi nama-nama tertentu.
- Medical AI yang kurang akurat untuk populasi kulit berwarna karena underrepresentation.
- Chatbot yang lebih "kasar" dalam bahasa non-Inggris karena kualitas RLHF data lebih rendah.

**Mitigasi teknis:**
- **Data filtering dan augmentation:** Aktif oversampling underrepresented groups.
- **Adversarial debiasing:** Training model adversarial untuk menghapus protected attributes dari representasi.
- **Fairness constraints:** Regularization term di training yang meminimalkan disparitas antara demographic groups.
- **Red-teaming:** Pengujian sistematis mencari bias sebelum deployment.
- **Diverse annotation:** RLHF dengan annotator beragam demografis.

### Keterbatasan Solusi Teknis

Solusi teknis saja tidak cukup:
1. **Whack-a-mole:** Menghapus satu bias sering memperkenalkan bias lain yang tidak terduga.
2. **Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure" — model belajar menipu metric fairness.
3. **Definisi bias bervariasi:** Apa yang "adil" bergantung pada konteks hukum, budaya, dan sosial yang berbeda-beda.

**Regulasi yang diperlukan:**
- EU AI Act: Wajibkan transparency untuk "high-risk" AI systems.
- Mandatory testing: Seperti crash tests untuk mobil, AI deployment publik butuh sertifikasi.
- Liability framework: Siapa yang bertanggung jawab jika AI menyebabkan kerugian?
- International coordination: Regulasi negara tunggal tidak efektif di era global AI.

---

## A4. Perbandingan RLHF vs DPO

### RLHF Pipeline Detail

**Tahap 1 — Supervised Fine-Tuning (SFT):**
Kumpulkan 10.000-100.000 prompt-respons berkualitas tinggi dari human demonstrators. Fine-tune LLM pre-trained menggunakan cross-entropy loss:

$$\mathcal{L}_\text{SFT} = -\sum_t \log \pi_\theta(y_t \mid x, y_{<t})$$

Hasilnya: model yang memahami format respons yang diinginkan.

**Tahap 2 — Reward Model Training:**
Generate K respons dari SFT model untuk setiap prompt. Human annotator meranking (K choose 2 pasangan). Train reward model r_φ menggunakan Bradley-Terry loss:

$$\mathcal{L}_\text{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

**Tahap 3 — RL Fine-Tuning dengan PPO:**
Optimasi LLM policy menggunakan reward model, dengan KL penalty:

$$\mathcal{L}_\text{RL} = \mathbb{E}[r_\phi(x, y)] - \beta \cdot \text{KL}[\pi_\theta(y|x) \| \pi_\text{ref}(y|x)]$$

PPO menyelesaikan ini sebagai RL problem: policy = LLM, action = token, reward = r_φ.

### DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) menunjukkan bahwa optimal policy di bawah KL-regularized objective memiliki bentuk closed-form:

$$\pi^*(y|x) \propto \pi_\text{ref}(y|x) \exp(r^*(x, y) / \beta)$$

Dengan rearranging, reward bisa diekspresikan sebagai:

$$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

Substitusi ke Bradley-Terry loss mengeliminasi reward model, langsung mengoptimasi policy:

$$\mathcal{L}_\text{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

### Perbandingan RLHF vs DPO

| Kriteria | RLHF | DPO |
|---|---|---|
| Jumlah model yang dibutuhkan | 3-4 (LLM + RM + RL + reference) | 2 (LLM + reference) |
| Kompleksitas training | Tinggi: butuh online sampling, PPO hypertuning | Rendah: supervised-like training |
| Stabilitas | Sering tidak stabil, RL notoriously tricky | Lebih stabil |
| Data requirements | Preferensi + demonstrasi | Hanya preferensi pairs |
| Compute overhead | 2-4× lebih berat dari SFT | ~1.2× lebih berat dari SFT |
| Performa empiris | Sangat baik (digunakan ChatGPT, Claude) | Matches atau sedikit di bawah RLHF pada banyak task |
| Reward generalization | Reward model bisa generalize ke prompts baru | Tidak ada explicit reward model |

**Referensi empiris:** Rafailov et al. (2023) menunjukkan DPO mengalahkan RLHF pada summarization (TL;DR) dan single-turn dialogue tasks, sambil 2-4× lebih efisien dalam compute.

### Rekomendasi untuk Low-Resource Setting

**Pilih DPO** untuk resource terbatas karena:
1. Tidak perlu train dan maintain reward model terpisah
2. Tidak butuh infrastruktur RL (rollouts, PPO, reward clipping)
3. Dataset preferensi saja sudah cukup — bisa gunakan dataset publik (HH-RLHF, OpenAssistant)
4. Debug lebih mudah: jika ada masalah, hanya satu loss function yang perlu dilihat
5. Memory-friendly: hanya 2 model copies di memory (current + reference)

Setup minimum yang layak: DPO dengan 1B model + 10K preference pairs + 1 GPU = tunable dalam 1-2 hari.

---

## A5. Perbandingan RAG vs RLM

### Kasus: Mencari Paper yang Saling Kontradiksi dalam 500 Research Papers

**Standard RAG:**
- Embedding search akan menemukan paper yang relevan dengan query "transformer scaling laws"
- Tapi RAG tidak bisa otomatis membandingkan semua pasangan paper untuk menemukan kontradiksi
- Untuk 500 paper, butuh multi-hop retrieval yang sangat kompleks
- Retrieval similarity ≠ factual comparison — dua paper bisa sama-sama relevan tapi berbeda argumentasi

**RLM (Recursive Language Model):**
- LLM bisa menulis Python code: `for p1, p2 in combinations(papers, 2): compare(p1, p2)`
- Bisa spawn sub-LLM calls untuk setiap pasangan yang teridentifikasi
- Bisa filter, aggregate, dan synthesize temuan kontradiksi secara programmatik
- Bisa handle seluruh 500 paper meskipun di luar context window model

**Kesimpulan:** RLM jauh lebih cocok untuk task agregasi dan perbandingan kompleks ini.

### Safety Risks dari LLM yang Mengeksekusi Python Code

1. **Sandbox escape:** Code bisa mencoba mengakses filesystem (`os.listdir('/')`) atau network
2. **Resource exhaustion:** Infinite loop atau alokasi memori besar menghabiskan resources
3. **Data exfiltration:** Code bisa mencoba mengirim data ke URL eksternal
4. **Privilege escalation:** Code bisa mencoba eksploitasi sistem

**Mitigasi sandboxing:**
- **Container isolation:** Docker container dengan network disabled, read-only filesystem
- **Resource limits:** `ulimit` untuk CPU time, memory, file size
- **Syscall filtering:** seccomp/gVisor untuk memblokir syscall berbahaya
- **Library whitelisting:** Hanya izinkan import dari safe list (numpy, pandas, sklearn)
- **Timeout:** Hard timeout untuk setiap code execution (misal 30 detik)
- **Human-in-the-loop:** Untuk code yang mengakses data sensitif, butuh approval

### Benchmark RLM (Zhang et al., arXiv:2512.24601)

**Benchmark yang digunakan:** OOLONG (132k token context), BrowseComp-Plus (1000 documents), dan custom aggregation tasks.

**Performance gap:** RLM(GPT-4o-mini) mengalahkan GPT-4o sebesar ~33% di OOLONG yang membutuhkan agregasi atas hampir setiap baris dataset besar. Pada input 10M+ token, RLM mempertahankan performa kuat sementara semua pendekatan standar lainnya gagal total.

### Kapan Standard RAG Masih Lebih Baik dari RLM?

1. **Latency-critical applications:** RAG hanya satu retrieval + satu generation; RLM butuh multiple LLM calls (5-20×+ lebih lambat)
2. **Simple lookup tasks:** "Apa definisi X dalam dokumen ini?" — RAG langsung dan murah
3. **Cost-sensitive:** RLM jauh lebih mahal karena multiple LLM invocations
4. **Structured KB:** Jika knowledge tersimpan di vector DB yang sudah terorganisir, RAG sangat efisien
5. **Low complexity:** Task yang bisa diselesaikan dengan 3-5 retrieved chunks tidak perlu overhead RLM

### Self-RAG vs Standard RAG vs RLM

| Aspek | Standard RAG | Self-RAG | RLM |
|---|---|---|---|
| Kapan retrieve? | Selalu | Model memutuskan sendiri | Programmatik, saat dibutuhkan |
| Context handling | Fixed top-k chunks | Adaptive selection | Programmatik exploration |
| Multi-hop | Manual pipeline | Partial support | Alami melalui recursion |
| Complexity | Rendah | Sedang | Tinggi |
| Cost | Rendah | Sedang | Tinggi |

**Pilih Self-RAG ketika:** Query bervariasi — beberapa butuh retrieval, beberapa tidak; ingin menghemat biaya vs always-retrieve.

---

## A6. Future of LLM and Generative AI

### Perkembangan Terbaru dari Google dan DeepSeek

**Google Gemini 2.5 Pro/Flash (2025-2026):**
- Context window sangat besar (hingga 2 juta token) untuk analisis dokumen skala besar
- Multimodal native: teks, gambar, audio, video dalam satu model
- "Deep Think" reasoning mode dengan chain-of-thought yang lebih panjang
- Peningkatan significant di coding benchmark (SWE-bench)

**DeepSeek-R1 dan DeepSeek-V3 (2025):**
- Mixture-of-Experts (MoE) architecture yang sangat efisien: hanya ~37B parameter aktif dari 671B total
- Training cost jauh lebih rendah dari GPT-4 kelas model berkat efisiensi arsitektur
- Reasoning capability mendekati frontier model dengan fraction dari biaya
- Open-source release yang mengejutkan industri, mendemokratisasi akses ke model kelas atas
- Multi-Token Prediction (MTP): memprediksi beberapa token sekaligus untuk inference lebih cepat

**Perkembangan umum (2025-2026):**
- Speculative decoding menjadi standar untuk mempercepat inference 2-3×
- Agentic reasoning: model dapat merencanakan dan mengeksekusi multi-step tasks secara otonom
- Test-time compute scaling: mengalokasikan lebih banyak compute saat inference untuk masalah sulit

### Apakah Tradeoff Context-Length vs Scalability Bisa Diselesaikan?

Saat ini terdapat tradeoff fundamental: attention complexity adalah O(n²) untuk context length n. Seiring meningkatnya context, compute dan memory tumbuh kuadratik.

**Pendekatan yang menjanjikan:**
1. **Linear Attention (Mamba, RWKV, Mamba-2):** Mengganti softmax attention dengan recurrence — O(n) complexity, tapi akurasi masih di bawah Transformer untuk banyak task
2. **Sparse Attention (Longformer, BigBird, FlashAttention-3):** Hanya attend ke subset token (local + global), efektif untuk banyak aplikasi
3. **External memory (RLM, Memorizing Transformer):** Menyimpan context di luar model seperti yang didemonstrasikan RLM paper
4. **Hierarchical processing:** Chunking dokumen dan memproses hierarkis

**Prediksi:** Dalam 3-5 tahun, efektif context yang bisa diakses model akan mendekati unlimited melalui kombinasi sparse attention + external memory + recursive processing. Namun "attention" kepada setiap token secara seragam mungkin tidak diperlukan — model yang cerdas bisa selectively attend.

### Apakah Arsitektur Saat Ini Bisa Mencapai AGI?

**Argumen bahwa arsitektur saat ini TIDAK cukup:**

1. **Grounding:** LLM beroperasi pada text token tanpa pengalaman embodied. Konsep seperti "panas", "berat", "rasa sakit" hanya dipahami statistikal, bukan eksperiensial.
2. **Causal reasoning:** Studi menunjukkan LLM sering gagal di novel causal reasoning yang membutuhkan pemahaman mekanistik dunia.
3. **Continual learning:** LLM tidak bisa belajar dari pengalaman baru tanpa fine-tuning penuh (catastrophic forgetting).
4. **Sample efficiency:** Manusia belajar dari ratusan contoh; LLM butuh miliaran. Ini mengindikasikan representasi yang fundamentally berbeda.
5. **World model:** LLM tidak memiliki explicit model tentang bagaimana dunia bekerja secara fisik.

**Argumen bahwa dengan modifikasi BISA:**

1. **Scale:** Mungkin ada fase transisi seperti GPT-2 → GPT-3 yang belum tercapai.
2. **Tool use + agency:** LLM + tools + memory + planning mungkin sufficient untuk banyak definisi AGI.
3. **Emergent capabilities:** Kemampuan baru muncul secara tak terduga di setiap order of magnitude scaling.
4. **Multimodal grounding:** Model yang belajar dari video, sensor, dan aksi (Gato, RT-2) mendekati embodied understanding.

**Kesimpulan:** Arsitektur Transformer saat ini, meski sangat powerful, kemungkinan tidak cukup *sendiri* untuk AGI level yang kuat (general problem-solving, novel scientific discovery, full common sense reasoning). Modifikasi yang dibutuhkan mencakup: persistent memory, continual learning, embodied training, explicit world models, dan mungkin fundamental architectural innovations. Namun "weak AGI" — sistem yang melampaui manusia di sebagian besar cognitive tasks berbasis teks — mungkin tercapai dengan iterasi arsitektur yang ada.

---

*References:*
1. Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023.
2. Ho et al. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
3. Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
4. Zhang et al. (2025). "Recursive Language Models." arXiv:2512.24601.
5. Asai et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." ICLR 2024.
6. Ouyang et al. (2022). "Training language models to follow instructions with human feedback." NeurIPS 2022.
7. Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
8. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
9. DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."
10. Google DeepMind (2025). "Gemini 2.5: Technical Report."