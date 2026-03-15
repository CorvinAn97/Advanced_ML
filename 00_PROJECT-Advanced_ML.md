# PROJECT-Advanced_ML

## ⚠️ WICHTIG: Lernplan gestartet am 15.03.!

**Aktueller Stand:** Sonntag, 15.03.2026, 09:30 Uhr
**Verbleibend:** 9 Tage bis zur Klausur (24.03.)

### HEUTE (Sonntag 15.03.) - START!
👉 Fange mit Transformers an (05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md)
👉 Fokus: Grobes Verständnis, nicht auswendig lernen!
👉 Notiere Fragen für Dozenten-Termin (Dienstag)

### Dateien sind jetzt nummeriert:
- 00_PROJECT-Advanced_ML.md (Hauptdokument)
- 01_VORBEREITUNG-Dienstag.md (Vorbereitung)
- 02_ZUSAMMENFASSUNG-00... (Inhaltsverzeichnis)
- 03-10... (Zusammenfassungen in Reihenfolge)

---

## Klausur-Info
- **Datum:** 24.03.2026
- **Zeit bis dahin:** 9 Tage (aktualisiert)
- **Stand:** 15.03.2026
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
- **Quelle:** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md
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
- **Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md
- **Seiten im Original:** ~36
- **Lernzeit empfohlen:** 2h
- **Kernpunkte:**
  - LSTM Gates (Forget, Input, Output)
  - Cell State Update
  - Vanishing/Exploding Gradients
  - Bahdanau Attention

#### 3. Word Embeddings
- **Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md
- **Seiten im Original:** ~20
- **Lernzeit empfohlen:** 1.5h
- **Kernpunkte:**
  - Word2Vec (CBOW, Skip-gram)
  - Distributional Hypothesis
  - FastText
  - BPE

#### 4. Q-Learning & Double DQN
- **Quelle:** 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md
- **Seiten im Original:** ~90
- **Lernzeit empfohlen:** 2.5h
- **Kernpunkte:**
  - Q-Learning Update
  - Bellman-Optimalitäts-Gleichung
  - Double DQN (Overestimation Problem)
  - Experience Replay
  - Target Networks

#### 5. GANs (Grundkonzept)
- **Quelle:** 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md
- **Seiten im Original:** ~40
- **Lernzeit empfohlen:** 1.5h
- **Kernpunkte:**
  - Generator & Discriminator
  - Minimax Loss
  - Mode Collapse
  - Conditional GAN

### 🟡 WICHTIG

#### 6. VAEs (Variational Autoencoder)
- **Quelle:** 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md
- **Seiten im Original:** ~20
- **Lernzeit empfohlen:** 1.5h
- **Kernpunkte:**
  - Reparametrisierungs-Trick
  - ELBO Loss
  - KL-Divergenz
  - Encoder-Decoder Struktur

#### 7. Diffusionsmodelle (Grundkonzept)
- **Quelle:** 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md
- **Seiten im Original:** ~30
- **Lernzeit empfohlen:** 2h
- **Kernpunkte:**
  - Forward Process (Noise hinzufügen)
  - Reverse Process (Denoising)
  - Training Loss
  - Classifier-Free Guidance

#### 8. Exploration in RL (UCB, Thompson Sampling)
- **Quelle:** 10_ZUSAMMENFASSUNG-08-RL-Teil-2.md
- **Seiten im Original:** ~35
- **Lernzeit empfohlen:** 1.5h
- **Kernpunkte:**
  - UCB Formel
  - Thompson Sampling Konzept
  - Exploration vs Exploitation
  - Pseudo-Counts (Konzept)

#### 9. XAI (LIME, SHAP)
- **Quelle:** 06_ZUSAMMENFASSUNG-04-XAI.md
- **Seiten im Original:** ~55
- **Lernzeit empfohlen:** 1.5h
- **Kernpunkte:**
  - LIME (lokale Approximation)
  - SHAP (Shapley Values)
  - Permutation Feature Importance

#### 10. Finetuning & PEFT (LoRA)
- **Quelle:** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md
- **Seiten im Original:** ~15
- **Lernzeit empfohlen:** 1h
- **Kernpunkte:**
  - LoRA (Low-Rank Adaptation)
  - Prompt Tuning
  - Catastrophic Forgetting

### 🟢 GRUNDWISSEN

#### 11. Imitation Learning (DAgger)
- **Quelle:** 08_ZUSAMMENFASSUNG-06-IL.md
- **Seiten im Original:** ~40
- **Lernzeit empfohlen:** 1h
- **Kernpunkte:**
  - Behavioral Cloning
  - Distributional Shift
  - DAgger Algorithmus

#### 12. RLHF & DPO
- **Quelle:** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md
- **Seiten im Original:** ~10
- **Lernzeit empfohlen:** 1h
- **Kernpunkte:**
  - RLHF Pipeline (Reward-Modell, PPO)
  - DPO (Direct Preference Optimization)
  - Reward Hacking

#### 13. Offline RL
- **Quelle:** 10_ZUSAMMENFASSUNG-08-RL-Teil-2.md
- **Seiten im Original:** ~20
- **Lernzeit empfohlen:** 1h
- **Kernpunkte:**
  - Distribution Shift Problem
  - Keine Online-Interaktion möglich

#### 14. XAI (Intrinsisch interpretierbare Modelle)
- **Quelle:** 06_ZUSAMMENFASSUNG-04-XAI.md
- **Seiten im Original:** ~15
- **Lernzeit empfohlen:** 0.5h
- **Kernpunkte:**
  - Lineare/logistische Regression
  - Entscheidungsbäume
  - Feature Importance

#### 15. Grundlagen ML (Backpropagation, Bias/Variance)
- **Quelle:** 03_ZUSAMMENFASSUNG-01-Wiederholung.md
- **Seiten im Original:** ~28
- **Lernzeit empfohlen:** 1h
- **Kernpunkte:**
  - Backpropagation
  - Bias vs Variance Tradeoff
  - Regularisierung

---

## 9-Tage Lernplan (aktualisiert)

### Tag 1 - SONNTAG 15.03. (HEUTE!) 🔥
**Thema:** Transformers & Self-Attention (Priorität 1)
- [ ] Lesen: 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md
- [ ] Original-PDF: AdvancedML-03-Transformers-LLMs.pdf (Seiten 1-30)
- [ ] Fokus: Intuition Self-Attention, Unterschied zu RNNs
- [ ] Notieren: Offene Fragen für Dozenten-Termin (Dienstag)
- [ ] Selbsttest: 5 Fragen aus PROJECT-File

### Tag 2 - MONTAG 16.03.
**Thema:** LSTM & RNNs + Vorbereitung Dozenten-Termin
- [ ] Lesen: 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md
- [ ] Fokus: LSTM Gates (forget, input, output), Seq2Seq
- [ ] Vorbereitung: Fragen für Dienstag notieren
- [ ] 01_VORBEREITUNG-Dienstag.md durchgehen

### Tag 3 - DIENSTAG 17.03. 👨‍🏫
**Thema:** DOZENTEN-TERMIN
- [ ] Vormittag: Letzte Fragen vorbereiten
- [ ] Termin: Fragen stellen, Notizen machen
- [ ] Nachmittag: Notizen einarbeiten, Lernplan anpassen

### Tag 4 - MITTWOCH 18.03.
**Thema:** Reinforcement Learning Teil 1
- [ ] Lesen: 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md
- [ ] Fokus: Q-Learning, DQN, Double DQN
- [ ] Original-PDF: AdvancedML-07-RL-Teil-1.pdf

### Tag 5 - DONNERSTAG 19.03.
**Thema:** Reinforcement Learning Teil 2
- [ ] Lesen: 10_ZUSAMMENFASSUNG-08-RL-Teil-2.md
- [ ] Fokus: Exploration, Offline RL
- [ ] Selbsttest: RL-Fragen

### Tag 6 - FREITAG 20.03.
**Thema:** GANs, VAEs, Diffusion
- [ ] Lesen: 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md
- [ ] Fokus: Unterschiede, wann welches Modell
- [ ] Original-PDF: AdvancedML-05-GANs-VAEs-Diffusion.pdf

### Tag 7 - SAMSTAG 21.03.
**Thema:** XAI + Imitation Learning
- [ ] Lesen: 06_ZUSAMMENFASSUNG-04-XAI.md
- [ ] Lesen: 08_ZUSAMMENFASSUNG-06-IL.md
- [ ] Fokus: LIME, SHAP (nur Grundideen!)

### Tag 8 - SONNTAG 22.03.
**Thema:** Wiederholung + Quiz
- [ ] Alle Zusammenfassungen überfliegen
- [ ] 30 Selbsttest-Fragen aus PROJECT-File
- [ ] Schwache Themen markieren

### Tag 9 - MONTAG 23.03. 🎯
**Thema:** LETZTE VORBEREITUNG
- [ ] Schwache Themen nochmal anschauen
- [ ] Formeln kurz wiederholen
- [ ] Fragen klären (falls noch offen)
- [ ] Früh ins Bett!

### Tag 10 - DIENSTAG 24.03. 📝
**KLAUSURTAG**
- [ ] Gutes Frühstück
- [ ] An Klausurteilnahme denken
- [ ] Erfolg!

---

## Querverweise

| Thema | Hauptquelle | Sekundärquelle |
|-------|-------------|----------------|
| Transformers | 05_ZUSAMMENFASSUNG-03 | 04_ZUSAMMENFASSUNG-02 (RNNs als Vergleich) |
| Word Embeddings | 04_ZUSAMMENFASSUNG-02 | 05_ZUSAMMENFASSUNG-03 (in LLMs verwendet) |
| LSTM | 04_ZUSAMMENFASSUNG-02 | - |
| Self-Attention | 05_ZUSAMMENFASSUNG-03 | 04_ZUSAMMENFASSUNG-02 (Bahdanau Attention als Vorläufer) |
| Q-Learning | 09_ZUSAMMENFASSUNG-07 | - |
| Double DQN | 09_ZUSAMMENFASSUNG-07 | - |
| Exploration | 10_ZUSAMMENFASSUNG-08 | 09_ZUSAMMENFASSUNG-07 (ε-greedy) |
| GANs | 07_ZUSAMMENFASSUNG-05 | - |
| VAEs | 07_ZUSAMMENFASSUNG-05 | - |
| Diffusion | 07_ZUSAMMENFASSUNG-05 | - |
| XAI | 06_ZUSAMMENFASSUNG-04 | - |
| RLHF | 05_ZUSAMMENFASSUNG-03 | - |

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
| 00 | [00_PROJECT-Advanced_ML.md](./00_PROJECT-Advanced_ML.md) | Hauptdokument & Lernplan |
| 01 | [01_VORBEREITUNG-Dienstag.md](./01_VORBEREITUNG-Dienstag.md) | Vorbereitung Dozenten-Termin |
| 02 | [02_ZUSAMMENFASSUNG-00-Inhaltsverzeichnis-Referenz.md](./02_ZUSAMMENFASSUNG-00-Inhaltsverzeichnis-Referenz.md) | Inhaltsverzeichnis & Relevanz |
| 03 | [03_ZUSAMMENFASSUNG-01-Wiederholung.md](./03_ZUSAMMENFASSUNG-01-Wiederholung.md) | ML-Grundlagen |
| 04 | [04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md](./04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md) | Word Embeddings & RNNs |
| 05 | [05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md](./05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md) | Transformers & LLMs |
| 06 | [06_ZUSAMMENFASSUNG-04-XAI.md](./06_ZUSAMMENFASSUNG-04-XAI.md) | Explainable AI |
| 07 | [07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md](./07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md) | GANs, VAEs, Diffusion |
| 08 | [08_ZUSAMMENFASSUNG-06-IL.md](./08_ZUSAMMENFASSUNG-06-IL.md) | Imitation Learning |
| 09 | [09_ZUSAMMENFASSUNG-07-RL-Teil-1.md](./09_ZUSAMMENFASSUNG-07-RL-Teil-1.md) | RL Teil 1 (Q-Learning, DQN) |
| 10 | [10_ZUSAMMENFASSUNG-08-RL-Teil-2.md](./10_ZUSAMMENFASSUNG-08-RL-Teil-2.md) | RL Teil 2 (Exploration, Offline RL) |

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
Aktualisiert: 15.03.2026 (Lernplan auf 9 Tage angepasst, Dateien nummeriert)
