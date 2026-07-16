
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
      const prompt = `<|user|>\n${userMessage}<|end|>\n<|assistant|>\n`;
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
        <Text style={styles.loadingText}>Loading AI Model...\n(~350MB, first launch only)</Text>
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
