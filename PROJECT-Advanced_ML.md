# PROJECT-Advanced_ML

## Klausur-Info
- **Datum:** 24.03.2026
- **Zeit bis dahin:** 10 Tage
- **Stand:** 14.03.2026
- **Erstellt von:** Sub-Agent David

---

## Nicht klausurrelevant (NICHT lernen!)

❌ **Konkrete Methoden der Tokenisierung** (byte pair encoding etc)
- Details nicht nötig, nur Konzept verstehen

❌ **Verschiedene Aktivierungsfunktionen**
- Nur ReLU, GELU, SwiGLU kennen, Details nicht

❌ **Genaue Funktionsweise von Positional Encodings**
- Zweck sollte bekannt sein, Formeln nicht auswendig lernen

❌ **Metriken & Benchmarks**
- GLUE, SuperGLUE, HELM - nicht klausurrelevant

❌ **Layer-wise relevance propagation**
- Nicht im Detail

❌ **Integrated Gradients**
- Nicht im Detail

❌ **Wasserstein GAN**
- Nicht klausurrelevant (laut Inhaltsverzeichnis)

❌ **Im RL-Teil:**
- DDPG, TRPO, PPO, SAC
- Implementierungstipps
- Information Gain in Deep RL (nur Konzept, keine Details)

---

## Klausurrelevante Themen (nach Wichtigkeit sortiert)

### 🔴 SEHR WICHTIG (wahrscheinlich in Klausur)

#### 1. Transformers & Self-Attention
- **Quelle:** ZUSAMMENFASSUNG-03-Transformers-LLMs.md
- **Seiten im Original:** ~80 (sehr ausführlich!)
- **Lernzeit empfohlen:** 3-4h
- **Kernpunkte:**
  - Self-Attention Mechanismus (Q, K, V)
  - Multi-Head Attention
  - Positional Encoding
  - Encoder vs Decoder
  - Masked Attention
  - Scaled Dot-Product Attention

#### 2. LSTM & RNNs
- **Quelle:** ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md
- **Seiten im Original:** ~36
- **Lernzeit empfohlen:** 2h
- **Kernpunkte:**
  - LSTM Gates (Forget, Input, Output)
  - Cell State Update
  - Vanishing/Exploding Gradients
  - Bahdanau Attention

#### 3. Word Embeddings
- **Quelle:** ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md
- **Seiten im Original:** ~20
- **Lernzeit empfohlen:** 1.5h
- **Kernpunkte:**
  - Word2Vec (CBOW, Skip-gram)
  - Distributional Hypothesis
  - FastText
  - BPE

#### 4. Q-Learning & Double DQN
- **Quelle:** ZUSAMMENFASSUNG-07-RL-Teil-1.md
- **Seiten im Original:** ~90
- **Lernzeit empfohlen:** 2.5h
- **Kernpunkte:**
  - Q-Learning Update
  - Bellman-Optimalitäts-Gleichung
  - Double DQN (Overestimation Problem)
  - Experience Replay
  - Target Networks

#### 5. GANs (Grundkonzept)
- **Quelle:** ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md
- **Seiten im Original:** ~40
- **Lernzeit empfohlen:** 1.5h
- **Kernpunkte:**
  - Generator & Discriminator
  - Minimax Loss
  - Mode Collapse
  - Conditional GAN

### 🟡 WICHTIG

#### 6. VAEs (Variational Autoencoder)
- **Quelle:** ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md
- **Seiten im Original:** ~20
- **Lernzeit empfohlen:** 1.5h
- **Kernpunkte:**
  - Reparametrisierungs-Trick
  - ELBO Loss
  - KL-Divergenz
  - Encoder-Decoder Struktur

#### 7. Diffusionsmodelle (Grundkonzept)
- **Quelle:** ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md
- **Seiten im Original:** ~30
- **Lernzeit empfohlen:** 2h
- **Kernpunkte:**
  - Forward Process (Noise hinzufügen)
  - Reverse Process (Denoising)
  - Training Loss
  - Classifier-Free Guidance

#### 8. Exploration in RL (UCB, Thompson Sampling)
- **Quelle:** ZUSAMMENFASSUNG-08-RL-Teil-2.md
- **Seiten im Original:** ~35
- **Lernzeit empfohlen:** 1.5h
- **Kernpunkte:**
  - UCB Formel
  - Thompson Sampling Konzept
  - Exploration vs Exploitation
  - Pseudo-Counts (Konzept)

#### 9. XAI (LIME, SHAP)
- **Quelle:** ZUSAMMENFASSUNG-04-XAI.md
- **Seiten im Original:** ~55
- **Lernzeit empfohlen:** 1.5h
- **Kernpunkte:**
  - LIME (lokale Approximation)
  - SHAP (Shapley Values)
  - Permutation Feature Importance

#### 10. Finetuning & PEFT (LoRA)
- **Quelle:** ZUSAMMENFASSUNG-03-Transformers-LLMs.md
- **Seiten im Original:** ~15
- **Lernzeit empfohlen:** 1h
- **Kernpunkte:**
  - LoRA (Low-Rank Adaptation)
  - Prompt Tuning
  - Catastrophic Forgetting

### 🟢 GRUNDWISSEN

#### 11. Imitation Learning (DAgger)
- **Quelle:** ZUSAMMENFASSUNG-06-IL.md
- **Seiten im Original:** ~40
- **Lernzeit empfohlen:** 1h
- **Kernpunkte:**
  - Behavioral Cloning
  - Distributional Shift
  - DAgger Algorithmus

#### 12. RLHF & DPO
- **Quelle:** ZUSAMMENFASSUNG-03-Transformers-LLMs.md
- **Seiten im Original:** ~10
- **Lernzeit empfohlen:** 1h
- **Kernpunkte:**
  - RLHF Pipeline (Reward-Modell, PPO)
  - DPO (Direct Preference Optimization)
  - Reward Hacking

#### 13. Offline RL
- **Quelle:** ZUSAMMENFASSUNG-08-RL-Teil-2.md
- **Seiten im Original:** ~20
- **Lernzeit empfohlen:** 1h
- **Kernpunkte:**
  - Distribution Shift Problem
  - Keine Online-Interaktion möglich

#### 14. XAI (Intrinsisch interpretierbare Modelle)
- **Quelle:** ZUSAMMENFASSUNG-04-XAI.md
- **Seiten im Original:** ~15
- **Lernzeit empfohlen:** 0.5h
- **Kernpunkte:**
  - Lineare/logistische Regression
  - Entscheidungsbäume
  - Feature Importance

#### 15. Grundlagen ML (Backpropagation, Bias/Variance)
- **Quelle:** ZUSAMMENFASSUNG-01-Wiederholung.md
- **Seiten im Original:** ~28
- **Lernzeit empfohlen:** 1h
- **Kernpunkte:**
  - Backpropagation
  - Bias vs Variance Tradeoff
  - Regularisierung

---

## 10-Tage Lernplan

### Tag 1 (14.03.): Transformers - Grundlagen
- [ ] Lesen: ZUSAMMENFASSUNG-03-Transformers-LLMs.md (Teil 1)
- [ ] Original-PDF: AdvancedML-03-Transformers-LLMs.pdf Seiten 1-40
- [ ] Selbsttest: Was ist Self-Attention? Erklären Sie Q, K, V!

### Tag 2 (15.03.): Transformers - Fortgeschritten
- [ ] Lesen: ZUSAMMENFASSUNG-03-Transformers-LLMs.md (Teil 2)
- [ ] Original-PDF: Seiten 41-80
- [ ] Selbsttest: Was ist der Unterschied zwischen Encoder und Decoder?

### Tag 3 (16.03.): LSTM & Word Embeddings
- [ ] Lesen: ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md
- [ ] Original-PDF: AdvancedML-02-WordEmbeddings-RNNs.pdf
- [ ] Fokus: LSTM Gates, Word2Vec, Bahdanau Attention
- [ ] Selbsttest: Erklären Sie die 3 LSTM Gates!

### Tag 4 (17.03.): Q-Learning & Double DQN
- [ ] Lesen: ZUSAMMENFASSUNG-07-RL-Teil-1.md
- [ ] Original-PDF: AdvancedML-07-RL-Teil-1.pdf (Q-Learning Teil)
- [ ] Fokus: Q-Learning Update, Target Networks, Double DQN
- [ ] Selbsttest: Was ist das Overestimation Problem?

### Tag 5 (18.03.): GANs & VAEs
- [ ] Lesen: ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md (GANs + VAEs)
- [ ] Original-PDF: AdvancedML-05-GANs-VAEs-Diffusion.pdf (Teil 1)
- [ ] Fokus: GAN Training, Reparametrisierungs-Trick, ELBO
- [ ] Selbsttest: Was ist Mode Collapse?

### Tag 6 (19.03.): Diffusionsmodelle
- [ ] Lesen: ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md (Diffusion)
- [ ] Original-PDF: AdvancedML-05-GANs-VAEs-Diffusion.pdf (Teil 2)
- [ ] Fokus: Forward/Reverse Process, Classifier-Free Guidance
- [ ] Selbsttest: Wie funktioniert DDPM Sampling?

### Tag 7 (20.03.): Exploration & Offline RL
- [ ] Lesen: ZUSAMMENFASSUNG-08-RL-Teil-2.md
- [ ] Original-PDF: AdvancedML-08-RL-Teil-2.pdf
- [ ] Fokus: UCB, Thompson Sampling, Offline RL Problem
- [ ] Selbsttest: Was ist der Unterschied zwischen UCB und Thompson Sampling?

### Tag 8 (21.03.): XAI & Imitation Learning
- [ ] Lesen: ZUSAMMENFASSUNG-04-XAI.md (LIME, SHAP)
- [ ] Lesen: ZUSAMMENFASSUNG-06-IL.md (DAgger)
- [ ] Fokus: LIME vs SHAP, Distributional Shift
- [ ] Selbsttest: Wie funktioniert LIME?

### Tag 9 (22.03.): Finetuning & Sonstiges
- [ ] Lesen: ZUSAMMENFASSUNG-03-Transformers-LLMs.md (PEFT, RLHF)
- [ ] Lesen: ZUSAMMENFASSUNG-01-Wiederholung.md (Grundlagen)
- [ ] Fokus: LoRA, DPO, Backpropagation
- [ ] Selbsttest: Was ist LoRA?

### Tag 10 (23.03.): Wiederholung & Quiz
- [ ] Durchsehen aller Zusammenfassungen
- [ ] Schwache Stellen nochmals anschauen
- [ ] Quiz durchführen (siehe unten)
- [ ] Früh schlafen gehen! 🛌

---

## Querverweise

| Thema | Hauptquelle | Sekundärquelle |
|-------|-------------|----------------|
| Transformers | ZUSAMMENFASSUNG-03 | ZUSAMMENFASSUNG-02 (RNNs als Vergleich) |
| Word Embeddings | ZUSAMMENFASSUNG-02 | ZUSAMMENFASSUNG-03 (in LLMs verwendet) |
| LSTM | ZUSAMMENFASSUNG-02 | - |
| Self-Attention | ZUSAMMENFASSUNG-03 | ZUSAMMENFASSUNG-02 (Bahdanau Attention als Vorläufer) |
| Q-Learning | ZUSAMMENFASSUNG-07 | - |
| Double DQN | ZUSAMMENFASSUNG-07 | - |
| Exploration | ZUSAMMENFASSUNG-08 | ZUSAMMENFASSUNG-07 (ε-greedy) |
| GANs | ZUSAMMENFASSUNG-05 | - |
| VAEs | ZUSAMMENFASSUNG-05 | - |
| Diffusion | ZUSAMMENFASSUNG-05 | - |
| XAI | ZUSAMMENFASSUNG-04 | - |
| RLHF | ZUSAMMENFASSUNG-03 | - |

---

## Offene Fragen/Unklarheiten

- ❓ **Double DQN:** Genauer Unterschied bei der Aktionsauswahl?
- ❓ **Thompson Sampling:** Wie genau wird die Posterior aktualisiert?
- ❓ **Diffusion:** Mathematische Herleitung der Sampling-Formel?
- ❓ **LoRA:** Warum funktioniert Low-Rank Adaption so gut?

---

## Zusammenfassungen Verzeichnis

| # | Datei | Thema |
|---|-------|-------|
| 00 | [ZUSAMMENFASSUNG-00-Inhaltsverzeichnis-Referenz.md](./ZUSAMMENFASSUNG-00-Inhaltsverzeichnis-Referenz.md) | Inhaltsverzeichnis & Relevanz |
| 01 | [ZUSAMMENFASSUNG-01-Wiederholung.md](./ZUSAMMENFASSUNG-01-Wiederholung.md) | ML-Grundlagen |
| 02 | [ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md](./ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md) | Word Embeddings & RNNs |
| 03 | [ZUSAMMENFASSUNG-03-Transformers-LLMs.md](./ZUSAMMENFASSUNG-03-Transformers-LLMs.md) | Transformers & LLMs |
| 04 | [ZUSAMMENFASSUNG-04-XAI.md](./ZUSAMMENFASSUNG-04-XAI.md) | Explainable AI |
| 05 | [ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md](./ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md) | GANs, VAEs, Diffusion |
| 06 | [ZUSAMMENFASSUNG-06-IL.md](./ZUSAMMENFASSUNG-06-IL.md) | Imitation Learning |
| 07 | [ZUSAMMENFASSUNG-07-RL-Teil-1.md](./ZUSAMMENFASSUNG-07-RL-Teil-1.md) | RL Teil 1 (Q-Learning, DQN) |
| 08 | [ZUSAMMENFASSUNG-08-RL-Teil-2.md](./ZUSAMMENFASSUNG-08-RL-Teil-2.md) | RL Teil 2 (Exploration, Offline RL) |

---

## Selbsttest-Fragen (Quiz)

### Transformers
1. Was ist Self-Attention? Erklären Sie Q, K, V!
2. Warum braucht man Positional Encoding?
3. Was ist der Unterschied zwischen Encoder und Decoder?
4. Wie funktioniert Multi-Head Attention?
5. Was ist Masked Attention und wozu dient sie?

### LSTM
6. Erklären Sie die 3 Gates im LSTM!
7. Was ist der Reparametrisierungs-Trick beim VAE?
8. Was ist der Unterschied zwischen LSTM und GRU?

### Word Embeddings
9. Was ist der Unterschied zwischen CBOW und Skip-gram?
10. Was ist die Distributional Hypothesis?

### RL
11. Was ist der Unterschied zwischen Q-Learning und SARSA?
12. Was ist das Overestimation Problem und wie löst Double DQN es?
13. Wozu dienen Target Networks?
14. Was ist der Unterschied zwischen On-Policy und Off-Policy?
15. Wie funktioniert UCB?
16. Was ist Thompson Sampling?
17. Was ist das Problem bei Offline RL?

### Generative Modelle
18. Was ist Mode Collapse bei GANs?
19. Was ist die ELBO beim VAE?
20. Erklären Sie Forward und Reverse Process bei Diffusionsmodellen!
21. Was ist Classifier-Free Guidance?

### XAI & IL
22. Wie funktioniert LIME?
23. Was sind Shapley Values?
24. Was ist Distributional Shift beim Imitation Learning?
25. Wie funktioniert DAgger?

### Sonstiges
26. Was ist LoRA?
27. Was ist der Unterschied zwischen RLHF und DPO?
28. Was ist Catastrophic Forgetting?
29. Was ist Bias vs Variance Tradeoff?
30. Erklären Sie Backpropagation!

---

## Tipps für die Klausur

✅ **Prioritäten:**
- Transformers und Q-Learning sind die wichtigsten Themen
- Konzepte verstehen, nicht alles auswendig lernen
- Formeln können abgeleitet werden, wenn Konzept klar

✅ **Typische Frage-Typen:**
- "Erklären Sie..." - Konzepte beschreiben
- "Was ist der Unterschied zwischen..." - Vergleiche
- "Wie funktioniert..." - Algorithmen beschreiben
- "Warum..." - Begründungen

✅ **Zeitmanagement:**
- Erst die einfachen Fragen lösen
- Bei Unsicherheit: Markieren und später zurückkommen
- Nicht zu lange an einer Frage hängen bleiben

✅ **Wichtige Formeln (auswendig):**
- Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Q-Learning Update: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
- UCB: a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]

---

## Letzte Änderung
Erstellt: 14.03.2026
