"""
Lab 4.5: Task-Specific Mobile LLM Deployment
=============================================
Objective: Fine-tune LLM kecil untuk domain spesifik dan deploy ke mobile
           menggunakan llama.cpp (GGUF format) + React Native atau Flutter.

PLATFORM NOTES:
- MacBook Air M3: Sangat cocok! llama.cpp punya dukungan Apple Metal native.
  Inferensi GGUF langsung di M3 sangat cepat.
- Workflow ini mostly command-line; lihat React Native section untuk mobile app.

INSTALL:
    pip install transformers peft torch  (sudah dari Lab 4.1)
    brew install cmake  (untuk build llama.cpp di macOS)

ATAU gunakan pre-built:
    pip install llama-cpp-python
"""

import os
import subprocess
import json

# ============================================================
# STEP 1: FINE-TUNE MODEL (dari Lab 4.1)
# ============================================================
"""
Gunakan kode dari lab4_1_lora_finetune.py dengan domain yang kamu pilih.
Contoh domain:
    - Medical Q&A: "Apa gejala diabetes tipe 2?"
    - University FAQ: "Bagaimana cara daftar sidang skripsi?"
    - Nutrition: "Berapa kalori nasi putih 100g?"
    - Quran & Hadith: "Apa keutamaan shalat tahajud?"
    - Indonesian Law: "Apa definisi tindak pidana ringan?"

Setelah fine-tuning, kamu punya folder: ./lora_finetuned_model/
"""

# ============================================================
# STEP 2: MERGE LoRA KE BASE MODEL
# ============================================================
def merge_lora_to_base(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    lora_adapter_path: str = "./lora_finetuned_model",
    output_path: str = "./merged_model"
):
    """Merge LoRA adapters ke base model untuk export ke GGUF."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[1/3] Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"[2/3] Loading LoRA adapter dari: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("[3/3] Merging dan unloading LoRA...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print(f"✓ Merged model disimpan ke: {output_path}")
    return output_path

# ============================================================
# STEP 3: KONVERSI KE GGUF FORMAT
# ============================================================
def convert_to_gguf(
    merged_model_path: str = "./merged_model",
    output_name: str = "model.gguf",
    quantization: str = "Q4_K_M"
):
    """
    Konversi model HuggingFace ke format GGUF untuk llama.cpp.

    Quantization options:
        Q2_K   → ~2.6 bit/weight, paling kecil, kualitas paling rendah
        Q4_0   → 4 bit, cepat
        Q4_K_M → 4 bit dengan mixed precision, rekomendasi (sweet spot)
        Q5_K_M → 5 bit, lebih akurat dari Q4_K_M
        Q8_0   → 8 bit, hampir sama dengan float16, paling besar

    Untuk mobile:
        Q4_K_M → ukuran file lebih kecil, kualitas baik → REKOMENDASI
        Q2_K   → ultra-kecil untuk device dengan storage terbatas
    """
    print("\n[GGUF Conversion]")

    # Install llama-cpp-python (include converter)
    print("Installing llama-cpp-python...")
    os.system("pip install llama-cpp-python --quiet")

    # Atau clone llama.cpp untuk full tools
    llama_cpp_dir = "./llama.cpp"
    if not os.path.exists(llama_cpp_dir):
        print("Cloning llama.cpp...")
        os.system(f"git clone https://github.com/ggerganov/llama.cpp {llama_cpp_dir}")

    # Build llama.cpp (M3: dengan Metal support)
    print("\nBuilding llama.cpp dengan Metal support (M3)...")
    build_cmd = f"cd {llama_cpp_dir} && cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j 4"
    print(f"  Command: {build_cmd}")
    # os.system(build_cmd)  # Uncomment untuk jalankan

    # Konversi HuggingFace → GGUF
    convert_cmd = f"""
    python {llama_cpp_dir}/convert_hf_to_gguf.py \\
        {merged_model_path} \\
        --outfile {output_name} \\
        --outtype f16
    """
    print(f"\n[Konversi HF → GGUF]:")
    print(f"  {convert_cmd.strip()}")

    # Quantize GGUF ke Q4_K_M
    quantize_cmd = f"""
    {llama_cpp_dir}/build/bin/llama-quantize \\
        {output_name} \\
        model_{quantization.lower()}.gguf \\
        {quantization}
    """
    print(f"\n[Quantize ke {quantization}]:")
    print(f"  {quantize_cmd.strip()}")

    print(f"\n✓ Output: model_{quantization.lower()}.gguf")
    print(f"  File size estimasi untuk 0.5B model:")
    print(f"    Q4_K_M → ~350MB (cocok untuk mobile!)")
    print(f"    Q2_K   → ~200MB (ultra-kecil)")
    return f"model_{quantization.lower()}.gguf"

# ============================================================
# STEP 4: TEST GGUF MODEL LOCALLY
# ============================================================
def test_gguf_model_local(gguf_path: str = "./model_q4_k_m.gguf"):
    """Test inferensi GGUF model menggunakan llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Install: pip install llama-cpp-python")
        print("Untuk M3 dengan Metal: CMAKE_ARGS='-DGGML_METAL=ON' pip install llama-cpp-python")
        return

    print(f"\n[Testing GGUF model: {gguf_path}]")

    llm = Llama(
        model_path=gguf_path,
        n_ctx=512,        # Context window
        n_batch=32,
        n_threads=8,      # Gunakan semua cores M3
        n_gpu_layers=-1,  # -1 = semua layer ke GPU (Metal di M3)
        verbose=False,
    )

    test_prompts = [
        "Apa manfaat makan sayuran hijau?",
        "Jelaskan cara kerja neural network secara singkat.",
    ]

    for prompt in test_prompts:
        print(f"\nQ: {prompt}")
        formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        output = llm(
            formatted,
            max_tokens=150,
            temperature=0.7,
            stop=["<|end|>", "<|user|>"],
        )
        print(f"A: {output['choices'][0]['text'].strip()}")
        print(f"   (Tokens: {output['usage']['completion_tokens']})")

# ============================================================
# STEP 5: REACT NATIVE APP (JavaScript)
# ============================================================
REACT_NATIVE_APP_JS = """
// ==========================================================
// React Native Mobile LLM App
// File: App.tsx
// ==========================================================
// Install: npm install react-native-llama-rn
//
// Setup:
//   1. Taruh file model_q4_k_m.gguf di android/app/src/main/assets/
//      atau ios/[AppName]/models/
//   2. npm install
//   3. npx react-native run-android  atau  run-ios

import React, { useState, useEffect, useRef } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, ScrollView,
  StyleSheet, KeyboardAvoidingView, Platform, ActivityIndicator,
} from 'react-native';
import { LlamaContext, initLlama } from 'react-native-llama-rn';

const MODEL_PATH = Platform.select({
  android: 'file:///android_asset/model_q4_k_m.gguf',
  ios: 'models/model_q4_k_m.gguf',
});

export default function App() {
  const [context, setContext] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  const scrollViewRef = useRef(null);

  // Load model saat app dibuka
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('Loading model from:', MODEL_PATH);
        const ctx = await initLlama({
          model: MODEL_PATH,
          n_ctx: 512,
          n_batch: 32,
          n_threads: 4,
          n_gpu_layers: Platform.OS === 'ios' ? 99 : 0,  // Metal di iOS
        });
        setContext(ctx);
        setModelLoading(false);
        console.log('Model loaded!');
      } catch (error) {
        console.error('Failed to load model:', error);
        setModelLoading(false);
      }
    };
    loadModel();
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || !context || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMessage }]);
    setLoading(true);

    try {
      const prompt = `<|user|>\\n${userMessage}<|end|>\\n<|assistant|>\\n`;
      let response = '';

      // Streaming generation
      await context.completion(
        {
          prompt,
          n_predict: 200,
          temperature: 0.7,
          top_p: 0.9,
          stop: ['<|end|>', '<|user|>'],
        },
        (data) => {
          response += data.token;
          // Update UI secara real-time (streaming)
          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last?.role === 'assistant') {
              return [...prev.slice(0, -1), { role: 'assistant', text: response }];
            }
            return [...prev, { role: 'assistant', text: data.token }];
          });
        }
      );
    } catch (error) {
      console.error('Inference error:', error);
      setMessages(prev => [...prev, { role: 'assistant', text: 'Error: ' + error.message }]);
    }

    setLoading(false);
  };

  if (modelLoading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Loading AI Model...\\n(~350MB, first launch only)</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>🤖 Offline AI Assistant</Text>
        <Text style={styles.headerSub}>Powered by on-device LLM</Text>
      </View>

      <ScrollView
        style={styles.messages}
        ref={scrollViewRef}
        onContentSizeChange={() => scrollViewRef.current?.scrollToEnd()}
      >
        {messages.map((msg, i) => (
          <View key={i} style={[styles.bubble, msg.role === 'user' ? styles.userBubble : styles.aiBubble]}>
            <Text style={msg.role === 'user' ? styles.userText : styles.aiText}>{msg.text}</Text>
          </View>
        ))}
        {loading && <ActivityIndicator style={{ margin: 10 }} />}
      </ScrollView>

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask anything..."
          multiline
          onSubmitEditing={sendMessage}
        />
        <TouchableOpacity style={styles.sendButton} onPress={sendMessage}>
          <Text style={styles.sendText}>Send</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  loadingText: { marginTop: 20, textAlign: 'center', color: '#666', fontSize: 16 },
  header: { backgroundColor: '#007AFF', padding: 20, paddingTop: 50, alignItems: 'center' },
  headerTitle: { color: '#FFF', fontSize: 20, fontWeight: 'bold' },
  headerSub: { color: 'rgba(255,255,255,0.8)', fontSize: 12, marginTop: 4 },
  messages: { flex: 1, padding: 16 },
  bubble: { maxWidth: '80%', padding: 12, borderRadius: 16, marginVertical: 4 },
  userBubble: { alignSelf: 'flex-end', backgroundColor: '#007AFF' },
  aiBubble: { alignSelf: 'flex-start', backgroundColor: '#FFF', borderWidth: 1, borderColor: '#E0E0E0' },
  userText: { color: '#FFF', fontSize: 15 },
  aiText: { color: '#333', fontSize: 15 },
  inputContainer: { flexDirection: 'row', padding: 12, backgroundColor: '#FFF', borderTopWidth: 1, borderTopColor: '#E0E0E0' },
  input: { flex: 1, borderWidth: 1, borderColor: '#DDD', borderRadius: 20, paddingHorizontal: 16, paddingVertical: 8, maxHeight: 100 },
  sendButton: { backgroundColor: '#007AFF', borderRadius: 20, paddingHorizontal: 20, paddingVertical: 10, marginLeft: 8, justifyContent: 'center' },
  sendText: { color: '#FFF', fontWeight: 'bold' },
});
"""

# ============================================================
# STEP 6: PERFORMANCE BENCHMARK
# ============================================================
def benchmark_quantization():
    """Benchmark berbagai quantization level untuk edge deployment."""
    print("\n" + "="*60)
    print("BENCHMARK: Quantization Comparison")
    print("="*60)

    # Estimasi berdasarkan model 1.5B parameter
    results = {
        "float32 (baseline)": {
            "size_mb": 6000,
            "vram_mb": 6000,
            "perplexity_increase": 0.0,
            "tokens_per_sec_m3": "~5",
            "mobile_feasible": "❌ Terlalu besar"
        },
        "float16": {
            "size_mb": 3000,
            "vram_mb": 3000,
            "perplexity_increase": "<0.1%",
            "tokens_per_sec_m3": "~15",
            "mobile_feasible": "❌ Borderline"
        },
        "Q8_0 (8-bit)": {
            "size_mb": 1600,
            "vram_mb": 1600,
            "perplexity_increase": "~0.1%",
            "tokens_per_sec_m3": "~25",
            "mobile_feasible": "✓ High-end phones"
        },
        "Q4_K_M (4-bit, REKOMENDASI)": {
            "size_mb": 900,
            "vram_mb": 900,
            "perplexity_increase": "~1-2%",
            "tokens_per_sec_m3": "~40",
            "mobile_feasible": "✓✓ Semua phones"
        },
        "Q2_K (2-bit)": {
            "size_mb": 500,
            "vram_mb": 500,
            "perplexity_increase": "~5-10%",
            "tokens_per_sec_m3": "~60",
            "mobile_feasible": "✓ Storage terbatas"
        },
    }

    print(f"\n{'Format':<30} {'Size':>8} {'PPL+':>8} {'Tok/s (M3)':>12} {'Mobile':>20}")
    print("-"*80)
    for fmt, r in results.items():
        print(f"{fmt:<30} {r['size_mb']:>7}MB {str(r['perplexity_increase']):>8} {r['tokens_per_sec_m3']:>12} {r['mobile_feasible']:>20}")

    print("\n→ REKOMENDASI untuk mobile: Q4_K_M (balance size/quality/speed)")
    print("→ Untuk safety-critical (medical/legal): Q8_0 atau float16 di server")

# ============================================================
# ACTIVITY ANSWERS
# ============================================================
def print_activity_answers():
    print("\n" + "="*60)
    print("ACTIVITY ANSWERS — Lab 4.5")
    print("="*60)

    answers = """
Q1: Trade-off 1B vs 3B model di mobile?
    
    1B Model:
    + Ukuran file kecil: Q4_K_M → ~600MB
    + Inference cepat: ~50-80 tokens/s di M3, ~15 tok/s di mid-range Android
    + RAM usage: ~800MB — aman untuk hampir semua smartphone modern
    - Kemampuan reasoning dan pengetahuan lebih terbatas
    
    3B Model:
    + Signifikan lebih pintar, understanding konteks lebih baik
    + Bisa handle instruksi kompleks dengan lebih baik
    - Ukuran: Q4_K_M → ~1.8GB
    - Lambat di low-end phones: ~5-10 tok/s di Android mid-range
    - RAM: ~2.5GB — bisa OOM di phones dengan 3GB RAM
    
    REKOMENDASI: 1B untuk low/mid-range phones, 3B untuk high-end only

---
Q2: Perbandingan 1-bit vs 4-bit vs 8-bit quantization
    
    1-bit (GPTQ/BitNet):
    - Ukuran: ekstrem kecil (~125MB untuk 1B model)
    - Perplexity drop: >15% — terasa jelas dalam percakapan
    - Masih eksperimental untuk deployment
    
    4-bit (Q4_K_M):
    - Ukuran: ~600MB untuk 1B model
    - Perplexity drop: ~1-2% — hampir tidak terasa oleh user
    - SWEET SPOT untuk mobile deployment
    
    8-bit (Q8_0):
    - Ukuran: ~1.2GB untuk 1B model
    - Perplexity drop: <0.1% — hampir identik float16
    - Cocok untuk server-side edge deployment (Raspberry Pi, dll)
    
    Benchmark perplexity (WikiText-2, Qwen2.5-1.5B):
    float16: ~8.1 | Q8_0: ~8.2 | Q4_K_M: ~8.5 | Q2_K: ~10.2

---
Q3: On-Device vs Cloud-Based LLM Inference
    
    On-Device:
    + PRIVASI: Data tidak keluar dari perangkat — kritis untuk medical/legal
    + Latensi: ~0ms network delay, first token langsung
    + Offline: Bisa berjalan tanpa internet
    + Cost: Gratis setelah download model
    - Kualitas: Terbatas model kecil (<7B), tidak bisa pakai GPT-4
    - Battery: Inferensi intensif menguras baterai
    - Storage: Model 600MB-2GB di device
    
    Cloud-Based:
    + Kualitas: Akses model terbaik (GPT-4, Claude 3.5, Gemini Ultra)
    + Always up-to-date
    - Privasi: Data dikirim ke server (regulasi HIPAA, GDPR perlu diperhatikan)
    - Latency: 500ms-2s per request, bergantung koneksi
    - Cost: $0.01-0.10 per 1000 tokens
    
    STRATEGI HYBRID: On-device untuk FAQ cepat/privat, cloud untuk pertanyaan
    kompleks yang butuh pengetahuan luas.

---
Q4: Safety measures untuk medical/religious Q&A chatbot
    
    1. DISCLAIMER yang jelas: "Ini bukan saran medis profesional."
       Tampilkan di setiap sesi, gunakan warna mencolok.
    
    2. Fact-checking layer: Untuk klaim medis kritis, tambahkan retrieval
       dari knowledge base terverifikasi (PubMed, WHO guidelines).
    
    3. Confidence scoring: Jika model tidak yakin, tampilkan peringatan
       dan sarankan konsultasi profesional.
    
    4. Hallucination detection: Cross-reference dengan RAG dari sumber
       terpercaya; jika ada kontradiksi, flag ke user.
    
    5. Hard boundaries: Daftar topik yang selalu dirujuk ke profesional
       (diagnosis penyakit serius, dosis obat, fatwa hukum besar).
    
    6. Audit logging: Log semua interaksi untuk review by professionals.
    
    7. Islamic Q&A: Sertakan sumber (hadith no., kitab) untuk verifikasi.
"""
    print(answers)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LAB 4.5: Task-Specific Mobile LLM")
    print("=" * 60)

    # Tampilkan workflow overview
    print("""
WORKFLOW LENGKAP:
=================
1. Fine-tune model   → python lab4_1_lora_finetune.py
2. Merge LoRA        → merge_lora_to_base() di script ini
3. Konversi ke GGUF  → convert_to_gguf()
4. Test lokal        → test_gguf_model_local()
5. Build React Native App → gunakan kode di REACT_NATIVE_APP_JS
6. Deploy ke device  → npx react-native run-android / run-ios
""")

    # Benchmark quantization
    benchmark_quantization()

    # Tampilkan React Native code
    react_native_output = "./react_native_app.tsx"
    with open(react_native_output, "w") as f:
        f.write(REACT_NATIVE_APP_JS)
    print(f"\n✓ React Native app code disimpan ke: {react_native_output}")

    # Activity answers
    print_activity_answers()

    print("\n✓ Lab 4.5 selesai!")
    print("\nCATATAN: Untuk deploy aktual:")
    print("  1. Jalankan merge_lora_to_base() setelah fine-tuning")
    print("  2. Jalankan convert_to_gguf() untuk konversi")
    print("  3. Salin GGUF ke folder assets React Native")
    print("  4. Gunakan kode dari react_native_app.tsx")