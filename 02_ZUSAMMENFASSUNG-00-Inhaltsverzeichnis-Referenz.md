# ZUSAMMENFASSUNG 00: Inhaltsverzeichnis & Klausurrelevanz

## Übersicht
- Seitenzahl: 2
- Hauptthemen: Klausurrelevante vs. nicht-klausurrelevante Themen, Gesamtstruktur der Vorlesung

## Detaillierte Inhalte

### 1. Nicht klausurrelevant (❌ NICHT lernen!)
- Konkrete Methoden der Tokenisierung (byte pair encoding etc)
- Verschiedene Aktivierungsfunktionen (Details)
- Genaue Funktionsweise von Positional Encodings (aber Zweck sollte bekannt sein)
- Metriken & Benchmarks
- Layer-wise relevance propagation
- Integrated Gradients
- Wasserstein GAN
- Im RL-Teil:
  - DDPG, TRPO, PPO, SAC
  - Implementierungstipps
  - Information Gain in Deep RL

### 2. Grobes Inhaltsverzeichnis der Vorlesung

#### 1. Darstellungsmöglichkeiten von Text
- **a. Bag of Words, TF-IDF** ✅ klausurrelevant
- **b. Word Embeddings** ✅ klausurrelevant

#### 2. Recurrent Neural Networks
- **a. Vanilla RNN** ✅ klausurrelevant
- **b. LSTM** ✅ klausurrelevant
- **c. Seq2Seq & Encoder-Decoder Architektur** ✅ klausurrelevant
- **d. Beam Search** ✅ klausurrelevant
- **e. Bahdanau-Attention (für seq2seq)** ✅ klausurrelevant

#### 3. Transformer Modelle und LLMs
- **a. Self-Attention und Cross-Attention** ✅ klausurrelevant
- **b. Embeddings** ✅ klausurrelevant
- **c. Encoder-only und Decoder-only Transformer** ✅ klausurrelevant
- **d. Finetuning** ✅ klausurrelevant
- **e. Parameter-Efficient Finetuning (PEFT)** ✅ klausurrelevant
- **f. RLHF (grundsätzliches Vorgehen)** ✅ klausurrelevant
- **g. DPO (Grundideen und Unterschied zu RLHF)** ✅ klausurrelevant
- **h. Reasoning in Sprachmodellen** ✅ klausurrelevant
- **i. Interaktion mit externen Datenbanken** ✅ klausurrelevant

#### 4. Explainable AI
- **a. Permutation Feature Importance** ✅ klausurrelevant
- **b. LIME (Grundidee)** ✅ klausurrelevant
- **c. SHAP (Grundidee)** ✅ klausurrelevant
- **d. Counterfactual Explanations** ✅ klausurrelevant

#### 5. Generative Bildmodelle
- **a. GANs** ✅ klausurrelevant
  - Grundideen, Loss von Diskriminator vs Loss von Generator
  - Problem von Mode Collapse kennen
  - Conditional Generation
  - Controllable Generation
  - Fidelity vs Diversity
- **b. VAE** ✅ klausurrelevant
  - Autoencoder - Warum kann dieser nicht sinnvoll zum Generieren verwendet werden?
  - VAE Konzept
- **c. Diffusionsmodelle** ✅ klausurrelevant
  - Grundkonzept: Forward-Process, Reverse-Process, Sampling
- **d. Vor- und Nachteile der verschiedenen generativen Bildmodelle** ✅ klausurrelevant
- **e. Grundkonzept von Latent Diffusion** ✅ klausurrelevant

#### 6. Imitation Learning / Behavioral Cloning
- ✅ klausurrelevant

#### 7. Reinforcement Learning
- **a. Wdh., insbesondere Q-Learning** ✅ klausurrelevant
- **b. DQN, Double DQN (welches Problem löst Double DQN?)** ✅ klausurrelevant
- **c. Policy Gradients** ✅ klausurrelevant
- **d. Actor-Critic** ✅ klausurrelevant
- **e. Exploration (Grundideen)** ✅ klausurrelevant
- **f. Offline RL (Schwierigkeit, Lösungsansätze)** ✅ klausurrelevant

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)
- ✅ **Transformers & Self-Attention** - Kern der modernen NLP-Architektur
- ✅ **GANs & Diffusionsmodelle** - Generative Modelle sehr wichtig
- ✅ **RL-Grundlagen (Q-Learning, DQN)** - Fundamentales Verständnis nötig
- ✅ **Word Embeddings** - Basis für alle NLP-Aufgaben

## Formeln/Algorithmen (falls vorhanden)
- Keine spezifischen Formeln im Inhaltsverzeichnis

## Eigene Notizen/Verständnis
- **Wichtige Erkenntnis:** Die Vorlesung ist in 7 Hauptblöcke unterteilt
- **Schwerpunkt:** ~60% Maschinelles Lernen, ~40% Reinforcement Learning
- **Transformers (Block 3)** sind der umfangreichste und wahrscheinlich wichtigste Teil
- **GANs/VAEs/Diffusion (Block 5)** sind prüfungsrelevant aber weniger detailliert
