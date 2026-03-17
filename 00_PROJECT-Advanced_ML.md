# PROJECT-Advanced_ML – Aktualisierte Übersicht

**Stand:** 17.03.2026, 16:42 Uhr  
**Klausurdatum:** 24.03.2026 (7 Tage verbleibend)  
**Status:** ✅ Alle Zusammenfassungen vollständig erstellt

---

## 1. PROJEKT-STATUS (ÜBERSICHT)

### 1.1 Alle 8 Zusammenfassungen – Status

| # | Datei | Thema | Status | Umfang | PDF-Abdeckung |
|---|-------|-------|--------|--------|---------------|
| 03 | 03_ZUSAMMENFASSUNG-01-Wiederholung.md | ML-Grundlagen | ✅ Vollständig | ~28 Seiten | 100% |
| 04 | 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md | Word Embeddings & RNNs | ✅ Vollständig | ~36 Seiten | 95% |
| 05 | 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md | Transformers & LLMs | ✅ Vollständig | ~80 Seiten | 100% |
| 06 | 06_ZUSAMMENFASSUNG-04-XAI.md | Explainable AI | ✅ Vollständig | ~55 Seiten | 100% |
| 07 | 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md | Generative Modelle | ✅ Vollständig | ~90 Seiten | 100% |
| 08 | 08_ZUSAMMENFASSUNG-06-IL.md | Imitation Learning | ✅ Vollständig | ~40 Seiten | 100% |
| 09 | 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md | RL Teil 1 (Q-Learning, DQN) | ✅ Vollständig | ~90 Seiten | 100% |
| 10 | 10_ZUSAMMENFASSUNG-08-RL-Teil-2.md | RL Teil 2 (Exploration, Offline RL) | ✅ Vollständig | ~76 Seiten | 100% |

**Gesamtumfang:** ~495 Seiten Zusammenfassungen  
**PDF-Abdeckung:** ~99% aller klausurrelevanten Themen

---

### 1.2 Zusätzliche Dateien

| Datei | Zweck | Status |
|-------|-------|--------|
| ANTWORTEN-Alle-32-Klausurfragen.md | Alle 32 Klausurfragen mit Antworten | ✅ Vollständig (100% Abdeckung) |
| CHEAT-SHEET-Klausur-24-03.md | Doppelseitige Formelsammlung | ✅ Vollständig |
| LERNPLAN-7-Tage-17-03-bis-24-03.md | Detaillierter 7-Tage-Lernplan | ✅ Vollständig |
| 01_VORBEREITUNG-Dienstag.md | Dozenten-Termin Vorbereitung | ✅ Vollständig |
| 02_ZUSAMMENFASSUNG-00-Inhaltsverzeichnis-Referenz.md | Klausurrelevanz-Übersicht | ✅ Vollständig |

---

## 2. INHALTSVERZEICHNIS ALLER ZUSAMMENFASSUNGEN

### 2.1 03_ZUSAMMENFASSUNG-01-Wiederholung.md (ML-Grundlagen)

**Hauptthemen:**
- Matrix-Notation und Matrix-Multiplikation (Zeile auf Spalte)
- Partielle Ableitungen und Gradient
- Supervised Learning (Trainingsdaten, Hypothese, Cost Function)
- Gradient Descent mit Lernrate α
- Bias vs Variance (Underfitting vs Overfitting)
- Regularisierung (λ-Einfluss)
- Fully Connected Networks (MLP)
- Backpropagation (Kettenregel rückwärts)
- Aktivierungsfunktionen (Sigmoid, tanh, ReLU, Softmax)

**Wichtige Formeln:**
```
Matrix-Multiplikation: C_ij = Σ_k A_ik · M_kj
Gradient Descent: θ ← θ - α · ∇_θ J(θ)
Least Squares: J(θ) = (1/2n) · Σ(h_θ(x^(i)) - y^(i))²
Regularized Cost: J(θ) = (1/2n) · Σ(h_θ(x^(i)) - y^(i))² + λ · Σθ_j²
Backpropagation: δ_j^(l) = Σ_i δ_i^(l+1) · Θ_ij^(l) · g'(z_j^(l))
Sigmoid: σ(x) = 1/(1+e^(-x)), σ'(x) = σ(x)·(1-σ(x))
```

**Klausurrelevanz:** 🟢 GRUNDWISSEN – Basis für alle weiteren Themen

---

### 2.2 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

**Hauptthemen:**
- Bag of Words (BoW) und TF-IDF
- Distributional Hypothesis ("You shall know a word by the company it keeps")
- Word2Vec (CBOW vs Skip-gram)
- FastText (Character n-grams, OOV-Behandlung)
- Byte Pair Encoding (BPE) Algorithmus
- Vanilla RNN und Vanishing Gradient Problem
- LSTM (3 Gates: Forget, Input, Output)
- GRU (2 Gates: Update, Reset)
- Bidirektionale RNNs/LSTMs
- Seq2Seq (Encoder-Decoder)
- Bahdanau Attention Mechanismus
- Beam Search

**Wichtige Formeln:**
```
TF-IDF: TF-IDF(w,d,D) = TF(w,d) × log(|D|/|{d∈D: w∈d}|)
RNN: h_t = tanh(W_h · h_{t-1} + W_x · x_t + b_h)
LSTM Gates:
  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
  C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
  h_t = o_t ⊙ tanh(C_t)
LSTM Parameter: N = 4 × d_h × (d_x + d_h + 1)
Attention: α_{t,i} = softmax(score(s_{t-1}, h_i)), c_t = Σ_i α_{t,i} · h_i
```

**Klausurrelevanz:** 🔴 SEHR HOCH – Grundlagen für alle NLP-Modelle

---

### 2.3 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md

**Hauptthemen:**
- Self-Attention (Query, Key, Value)
- Multi-Head Attention
- Positional Encoding (Sinus/Cosinus, RoPE)
- Encoder vs Decoder Architektur
- Masked Attention (Causal Masking)
- Layer Normalization (Pre-Norm vs Post-Norm)
- Feed Forward Networks (GELU, SwiGLU)
- Tokenisierung (BPE, WordPiece, SentencePiece)
- Encoder-only vs Decoder-only Modelle
- Finetuning (Full vs PEFT, LoRA, QLoRA)
- RLHF (Reward-Modell, PPO)
- DPO (Direct Preference Optimization)
- Constitutional AI
- Reasoning (Chain-of-Thought, Tree of Thoughts)
- RAG (Retrieval-Augmented Generation)

**Wichtige Formeln:**
```
Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k) · V
Multi-Head: MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
Positional Encoding: PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d))
Layer Norm: LN(x) = γ ⊙ (x-μ)/√(σ²+ε) + β
LoRA: W' = W + B·A, wobei B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
Reward Loss: L = -E[log σ(R(x,y_w) - R(x,y_l))]
DPO: L_DPO = -E[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

**Klausurrelevanz:** 🔴 SEHR HOCH – Kern der modernen NLP-Architektur

---

### 2.4 06_ZUSAMMENFASSUNG-04-XAI.md

**Hauptthemen:**
- Interpretable ML vs Explainable AI
- Intrinsisch vs Post-hoc Interpretierbarkeit
- Lokal vs Global Erklärungen
- Lineare/logistische Regression (intrinsisch interpretierbar)
- Entscheidungsbäume und Feature Importance
- Permutation Feature Importance (PFI)
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Counterfactual Explanations
- Layer-wise Relevance Propagation (LRP)
- Integrated Gradients (IG)

**Wichtige Formeln:**
```
Lineare Regression: y = w₀ + w₁x₁ + ... + w_dx_d
Logistische Regression: p = 1/(1+e^(-(w₀+w₁x₁+...+w_dx_d)))
LIME: g = argmin_{g∈G} L(f,g,π_x) + Ω(g)
Shapley Value: φ_i = Σ_{S⊆N\{i}} (|S|!(|N|-|S|-1)!/|N|!) × [v(S∪{i}) - v(S)]
Integrated Gradients: IG_i(x) = (x_i-x'_i) × ∫₀¹ ∂F(x'+α(x-x'))/∂x_i dα
```

**Klausurrelevanz:** 🟡 WICHTIG – Grundkonzepte verstehen

---

### 2.5 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md

**Hauptthemen:**
- GAN Grundkonzept (Generator vs Discriminator)
- GAN Training (Binary Cross Entropy Loss)
- Mode Collapse Problem
- Wasserstein GAN (WGAN, Gradient Penalty)
- Conditional GAN
- Controllable Generation
- GAN Evaluation (Inception Score, FID)
- VAE (Variational Autoencoder)
- Reparametrisierungs-Trick
- ELBO (Evidence Lower Bound)
- Diffusionsmodelle (Forward/Reverse Process)
- DDPM Sampling
- DDIM (beschleunigtes Sampling)
- Classifier-Free Guidance
- Latent Diffusion (Stable Diffusion)

**Wichtige Formeln:**
```
GAN BCE Loss: L_D = -[log(D(x)) + log(1-D(G(z)))]
Wasserstein Loss: min_G max_C E[C(x)] - E[C(G(z))]
VAE ELBO: L = E[||x-d(z)||²] + D_KL(q(z|x) || p(z))
Reparametrisierung: z = μ + σ ⊙ ε, ε ~ N(0,I)
KL-Divergenz (Gauss): D_KL = 0.5 · Σ(μ² + σ² - log(σ²) - 1)
Diffusion Forward: x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε
Diffusion Training: L = ||ε - ε_θ(x_t,t)||²
Diffusion Sampling (DDPM): x_{t-1} = (x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ)/√α_t + σ_t·z
Classifier-Free Guidance: ε̂ = ε_θ(x_t,t,∅) + w·(ε_θ(x_t,t,c) - ε_θ(x_t,t,∅))
FID: FID = ||μ₁-μ₂||² + Tr(Σ₁+Σ₂-2·√(Σ₁·Σ₂))
```

**Klausurrelevanz:** 🔴 SEHR HOCH – Generative Modelle sehr wichtig

---

### 2.6 08_ZUSAMMENFASSUNG-06-IL.md

**Hauptthemen:**
- Behavioral Cloning (Grundkonzept)
- Distributional Shift Problem
- Fehlerakkumulation (O(εT²) statt O(εT))
- NVIDIA DRIVES Beispiel (Data Augmentation mit seitlichen Kameras)
- Goal-Conditioned Behavioral Cloning
- Automatisierte Datensammlung
- DAgger (Dataset Aggregation) Algorithmus
- Warum IL nicht genug ist (Vorteile von RL)

**Wichtige Formeln:**
```
BC Loss: L = -E_{(o,a)~D_demo}[log π_θ(a|o)]
Distributional Shift (Worst-Case): E[Σ_t c(s_t,a_t)] = O(εT²)
DAgger Algorithmus:
  for i = 1,2,...:
    D_i = rollout(π_i)
    D_i^labeled = expert_label(D_i)
    D = D ∪ D_i^labeled
    π_{i+1} = train(D)
Goal-Conditioned BC: L = -Σ_i Σ_t log π_θ(a_t^i | s_t^i, g_i=s_{T_i}^i)
```

**Klausurrelevanz:** 🟡 WICHTIG – DAgger und Distributional Shift wichtig

---

### 2.7 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md

**Hauptthemen:**
- RL-Grundlagen (Agent, Environment, State, Action, Reward)
- Policy π (deterministisch vs stochastisch)
- Return G_t (diskontierter kumulativer Reward)
- Value Functions (v_π(s), q_π(s,a))
- Bellman-Gleichungen (Expectation und Optimality)
- Exploration vs Exploitation
- Monte-Carlo Methods
- Temporal-Difference (TD) Learning
- SARSA (On-Policy TD Control)
- Q-Learning (Off-Policy TD Control)
- ε-Greedy Exploration
- Deep Q-Networks (DQN)
- Experience Replay
- Target Networks
- Double DQN (Overestimation Problem)
- Function Approximation
- On-Policy vs Off-Policy

**Wichtige Formeln:**
```
Return: G_t = Σ_{k=0}^{∞} γ^k · r_{t+k+1}
State-Value: v_π(s) = E_π[G_t | S_t = s]
Action-Value: q_π(s,a) = E_π[G_t | S_t = s, A_t = a]
Bellman-Optimality (q*): q*(s,a) = E[r + γ·max_{a'} q*(s',a')]
SARSA: Q(s,a) ← Q(s,a) + α·[r + γ·Q(s',a') - Q(s,a)]
Q-Learning: Q(s,a) ← Q(s,a) + α·[r + γ·max_{a'} Q(s',a') - Q(s,a)]
TD Error: δ_t = R_{t+1} + γ·V(S_{t+1}) - V(S_t)
DQN Loss: L(w) = E[(r + γ·max_{a'} Q_target(s',a';w⁻) - Q(s,a;w))²]
Double DQN: a* = argmax_{a'} Q(s',a';w), Q_double = r + γ·Q_target(s',a*;w⁻)
ε-Greedy: π(a|s) = 1-ε+ε/|A| wenn a=argmax Q, sonst ε/|A|
```

**Klausurrelevanz:** 🔴 SEHR HOCH – Q-Learning und DQN gehören zu den wichtigsten Themen

---

### 2.8 10_ZUSAMMENFASSUNG-08-RL-Teil-2.md

**Hauptthemen:**
- Warum Exploration schwierig ist (Montezuma's Revenge Beispiel)
- Exploration vs Exploitation Dilemma
- Bandits (Multi-armed Bandit Problem)
- Regret (Qualitätsmaß für Exploration)
- UCB (Upper Confidence Bound)
- Thompson Sampling
- Information Gain / Information-Directed Sampling
- Optimistische Exploration in Deep RL
- Pseudo-Counts (Zählen in großen Zustandsräumen)
- Bootstrapped DQN
- Offline Reinforcement Learning
- Distribution Shift Problem
- Counterfactual Queries
- Policy Constraint Methoden (BRAC, BEAR, BCQ, TD3+BC)
- Implicit Policy Constraints (AWR, AWAC, CRR)
- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL, Expectile Loss)
- Sequence Modelling (Decision Transformer, Trajectory Transformer)

**Wichtige Formeln:**
```
UCB: a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]
Thompson Sampling: θ ~ p̂(θ), a = argmax_a E_θ[r|a]
Regret: Reg(T) = T·E[r(a*)] - Σ_{t=1}^T E[r(a_t)]
Pseudo-Count: N̂(s) = p_θ(s)/(p_θ'(s) - p_θ(s))
Information Gain: IG(z,y|a) = E_y[H(p̂(z)) - H(p̂(z|y))|a]
Expectile Loss (IQL): l_τ(x) = (1-τ)|x|² wenn x>0, sonst τ|x|²
CQL Objective: L_CQL = α·E_{s,a~μ}[Q(s,a)] - α·E_{(s,a)~D}[Q(s,a)] + L_TD(Q)
Advantage-Weighted Policy: π*(a|s) = (1/Z(s))·π^D(a|s)·exp(A(s,a)/λ)
Decision Transformer Input: (R_1,s_1,a_1,R_2,s_2,a_2,...), R_t = Σ_{t'=t}^T r_{t'}
```

**Klausurrelevanz:** 🔴 SEHR WICHTIG – UCB, Thompson Sampling, Offline RL Problem

---

## 3. KLAUSUR-FRAGEN-ABDECKUNG (32 FRAGEN)

### 3.1 Vollständige Zuordnung aller 32 Fragen

| Frage # | Thema | Frage | Antwort in Datei | Zusammenfassung |
|---------|-------|-------|------------------|-----------------|
| 1 | Transformers | Self-Attention: Q,K,V? | ✅ beantwortet | 05_ZUSAMMENFASSUNG-03 |
| 2 | Transformers | Positional Encoding | ✅ beantwortet | 05_ZUSAMMENFASSUNG-03 |
| 3 | Transformers | Encoder vs Decoder | ✅ beantwortet | 05_ZUSAMMENFASSUNG-03 |
| 4 | Transformers | Multi-Head Attention | ✅ beantwortet | 05_ZUSAMMENFASSUNG-03 |
| 5 | Transformers | Masked Attention | ✅ beantwortet | 05_ZUSAMMENFASSUNG-03 |
| 6 | Transformers | Transformer > RNN | ✅ beantwortet | 05_ZUSAMMENFASSUNG-03, 04_ZUSAMMENFASSUNG-02 |
| 7 | Transformers | SwiGLU vs GELU | ✅ beantwortet | 05_ZUSAMMENFASSUNG-03 |
| 8 | LSTM/RNNs | 3 LSTM Gates | ✅ beantwortet | 04_ZUSAMMENFASSUNG-02 |
| 9 | LSTM/RNNs | Vanishing Gradient | ✅ beantwortet | 04_ZUSAMMENFASSUNG-02 |
| 10 | LSTM/RNNs | LSTM vs GRU | ✅ beantwortet | 04_ZUSAMMENFASSUNG-02 |
| 11 | LSTM/RNNs | Bidirectional LSTM | ✅ beantwortet | 04_ZUSAMMENFASSUNG-02 |
| 12 | Word Embeddings | CBOW vs Skip-gram | ✅ beantwortet | 04_ZUSAMMENFASSUNG-02 |
| 13 | Word Embeddings | Distributional Hypothesis | ✅ beantwortet | 04_ZUSAMMENFASSUNG-02 |
| 14 | Word Embeddings | FastText vs Word2Vec | ✅ beantwortet | 04_ZUSAMMENFASSUNG-02 |
| 15 | Word Embeddings | TF-IDF Formel | ✅ beantwortet | 04_ZUSAMMENFASSUNG-02 |
| 16 | Word Embeddings | BPE Algorithmus | ✅ beantwortet | 04_ZUSAMMENFASSUNG-02 |
| 17 | RL | Q-Learning Update | ✅ beantwortet | 09_ZUSAMMENFASSUNG-07 |
| 18 | RL | Q-Learning vs SARSA | ✅ beantwortet | 09_ZUSAMMENFASSUNG-07 |
| 19 | RL | Overestimation DQN | ✅ beantwortet | 09_ZUSAMMENFASSUNG-07 |
| 20 | RL | Target Networks | ✅ beantwortet | 09_ZUSAMMENFASSUNG-07 |
| 21 | RL | UCB Formel | ✅ beantwortet | 10_ZUSAMMENFASSUNG-08 |
| 22 | RL | Thompson Sampling | ✅ beantwortet | 10_ZUSAMMENFASSUNG-08 |
| 23 | RL | Bellman-Gleichung | ✅ beantwortet | 09_ZUSAMMENFASSUNG-07 |
| 24 | Generative Modelle | Mode Collapse GAN | ✅ beantwortet | 07_ZUSAMMENFASSUNG-05 |
| 25 | Generative Modelle | VAE Reparametrisierung | ✅ beantwortet | 07_ZUSAMMENFASSUNG-05 |
| 26 | Generative Modelle | Diffusion Forward/Reverse | ✅ beantwortet | 07_ZUSAMMENFASSUNG-05 |
| 27 | Generative Modelle | Classifier-Free Guidance | ✅ beantwortet | 07_ZUSAMMENFASSUNG-05 |
| 28 | XAI | LIME vs SHAP | ✅ beantwortet | 06_ZUSAMMENFASSUNG-04 |
| 29 | XAI | SHAP (Shapley Values) | ✅ beantwortet | 06_ZUSAMMENFASSUNG-04 |
| 30 | XAI | PFI bei Korrelation | ✅ beantwortet | 06_ZUSAMMENFASSUNG-04 |
| 31 | Imitation Learning | Distributional Shift BC | ✅ beantwortet | 08_ZUSAMMENFASSUNG-06 |
| 32 | Imitation Learning | DAgger Algorithmus | ✅ beantwortet | 08_ZUSAMMENFASSUNG-06 |

**Abdeckung:** 32/32 Fragen (100%) ✅

---

### 3.2 Fragenverteilung nach Themenbereich

| Themenbereich | Fragen | Datei | Status |
|---------------|--------|-------|--------|
| Transformers | 7 | 05_ZUSAMMENFASSUNG-03 | ✅ 100% |
| LSTM & RNNs | 4 | 04_ZUSAMMENFASSUNG-02 | ✅ 100% |
| Word Embeddings | 5 | 04_ZUSAMMENFASSUNG-02 | ✅ 100% |
| RL | 7 | 09_ZUSAMMENFASSUNG-07, 10_ZUSAMMENFASSUNG-08 | ✅ 100% |
| Generative Modelle | 4 | 07_ZUSAMMENFASSUNG-05 | ✅ 100% |
| XAI | 3 | 06_ZUSAMMENFASSUNG-04 | ✅ 100% |
| Imitation Learning | 2 | 08_ZUSAMMENFASSUNG-06 | ✅ 100% |

---

## 4. LERNPLAN-INTEGRATION (7 TAGE BIS 24.03)

### 4.1 Verknüpfung mit LERNPLAN-7-Tage-17-03-bis-24-03.md

**Aktueller Stand:** Dienstag, 17.03.2026 (Tag 1 des Lernplans)  
**Verbleibend:** 7 Tage bis Klausur (24.03.2026)

### 4.2 Prioritäten für verbleibende Tage

| Tag | Datum | Fokus | Priorität | Dateien |
|-----|-------|-------|-----------|---------|
| Tag 1 | Di 17.03. | Transformers & Self-Attention | 🔴 SEHR WICHTIG | 05_ZUSAMMENFASSUNG-03 |
| Tag 2 | Mi 18.03. | LSTM & Word Embeddings | 🔴 SEHR WICHTIG | 04_ZUSAMMENFASSUNG-02 |
| Tag 3 | Do 19.03. | Q-Learning & Double DQN | 🔴 SEHR WICHTIG | 09_ZUSAMMENFASSUNG-07 |
| Tag 4 | Fr 20.03. | GANs & VAEs | 🟡 WICHTIG | 07_ZUSAMMENFASSUNG-05 (Seiten 1-30) |
| Tag 5 | Sa 21.03. | Diffusion & XAI | 🟡 WICHTIG | 07_ZUSAMMENFASSUNG-05 (31-60), 06_ZUSAMMENFASSUNG-04 |
| Tag 6 | So 22.03. | RL Exploration & IL | 🟡 WICHTIG | 10_ZUSAMMENFASSUNG-08, 08_ZUSAMMENFASSUNG-06 |
| Tag 7 | Mo 23.03. | PUFFERTAG – Wiederholung | 🔴 WICHTIG | ALLE Zusammenfassungen |
| Tag 8 | Di 24.03. | **KLAUSURTAG** 📝 | - | - |

---

### 4.3 Empfohlene Lesereihenfolge (Priorisiert)

**Priorität 1 (Tag 1-3 – SEHR WICHTIG):**
1. 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md (80 Seiten, 🔴)
2. 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md (36 Seiten, 🔴)
3. 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md (90 Seiten, 🔴)

**Priorität 2 (Tag 4-5 – WICHTIG):**
4. 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md (90 Seiten, 🟡)
5. 06_ZUSAMMENFASSUNG-04-XAI.md (55 Seiten, 🟡)

**Priorität 3 (Tag 6 – GRUNDWISSEN):**
6. 10_ZUSAMMENFASSUNG-08-RL-Teil-2.md (76 Seiten, 🟡)
7. 08_ZUSAMMENFASSUNG-06-IL.md (40 Seiten, 🟢)
8. 03_ZUSAMMENFASSUNG-01-Wiederholung.md (28 Seiten, 🟢)

---

### 4.4 Wichtige Formeln (auswendig für Klausur)

**Must-Know Formeln (auswendig lernen!):**
```
1. Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k) · V
2. Q-Learning: Q(s,a) ← Q(s,a) + α·[r + γ·max_{a'} Q(s',a') - Q(s,a)]
3. UCB: a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]
4. LSTM Gates: f_t, i_t, C̃_t, C_t, o_t, h_t (alle 6 Gleichungen)
5. TF-IDF: TF-IDF = TF × log(|D|/|{d: w∈d}|)
6. ELBO (VAE): L = Reconstruction + KL-Divergenz
7. Shapley Value: φ_i = Σ_{S⊆N\{i}} (|S|!(|N|-|S|-1)!/|N|!) × [v(S∪{i}) - v(S)]
```

---

## 5. VERFÜGBARE RESSOURCEN

### 5.1 Alle erstellten Dateien mit Zweck

| Datei | Zweck | Empfohlene Nutzung |
|-------|-------|-------------------|
| **00_PROJECT-Advanced_ML.md** | Hauptdokument, Übersicht, Lernplan | 📌 ZENTRAL – Regelmäßig konsultieren |
| **01_VORBEREITUNG-Dienstag.md** | Dozenten-Termin Vorbereitung | ✅ Abgeschlossen (17.03. genutzt) |
| **02_ZUSAMMENFASSUNG-00-Inhaltsverzeichnis-Referenz.md** | Klausurrelevanz-Übersicht | 📋 Schnellreferenz für Relevanz |
| **03_ZUSAMMENFASSUNG-01-Wiederholung.md** | ML-Grundlagen | 🟢 GRUNDWISSEN – Bei Bedarf nachschlagen |
| **04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md** | Word Embeddings & RNNs | 🔴 SEHR WICHTIG – Tag 2 Fokus |
| **05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md** | Transformers & LLMs | 🔴 SEHR WICHTIG – Tag 1 Fokus |
| **06_ZUSAMMENFASSUNG-04-XAI.md** | Explainable AI | 🟡 WICHTIG – Tag 5 Fokus |
| **07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md** | Generative Modelle | 🔴 SEHR WICHTIG – Tag 4 Fokus |
| **08_ZUSAMMENFASSUNG-06-IL.md** | Imitation Learning | 🟡 WICHTIG – Tag 6 Fokus |
| **09_ZUSAMMENFASSUNG-07-RL-Teil-1.md** | RL Teil 1 | 🔴 SEHR WICHTIG – Tag 3 Fokus |
| **10_ZUSAMMENFASSUNG-08-RL-Teil-2.md** | RL Teil 2 | 🟡 WICHTIG – Tag 6 Fokus |
| **ANTWORTEN-Alle-32-Klausurfragen.md** | Alle 32 Fragen mit Antworten | ✅ QUIZ – Selbsttest durchführen |
| **CHEAT-SHEET-Klausur-24-03.md** | Doppelseitige Formelsammlung | 📝 KLAUSUR – Ausdruck für Klausur |
| **LERNPLAN-7-Tage-17-03-bis-24-03.md** | Detaillierter 7-Tage-Lernplan | 📅 TAGESPLAN – Täglich abarbeiten |

---

### 5.2 Empfohlene Lesereihenfolge für Corvin

**Effizientester Pfad (7 Tage):**

```
Tag 1 (17.03.):  05_ZUSAMMENFASSUNG-03 (Transformers)     → 2.5h morgens, 1.5h nachmittags, 1h abends
Tag 2 (18.03.):  04_ZUSAMMENFASSUNG-02 (Embeddings/RNNs)  → 2.5h morgens, 1.5h nachmittags, 1h abends
Tag 3 (19.03.):  09_ZUSAMMENFASSUNG-07 (RL-Teil-1)        → 2.5h morgens, 1.5h nachmittags, 1h abends
Tag 4 (20.03.):  07_ZUSAMMENFASSUNG-05 (GANs/VAEs)        → 2.5h morgens, 1.5h nachmittags, 1h abends
Tag 5 (21.03.):  07_ZUSAMMENFASSUNG-05 (Diffusion) + 06   → 2.5h morgens, 1.5h nachmittags, 1h abends
Tag 6 (22.03.):  10_ZUSAMMENFASSUNG-08 + 08_ZUSAMMENFASSUNG-06 → 2.5h morgens, 1.5h nachmittags, 1h abends
Tag 7 (23.03.):  ALLE überfliegen + 30 Selbsttest-Fragen  → 2h morgens, 2h nachmittags, 1h abends
Tag 8 (24.03.):  KLAUSUR 📝
```

**Tipp:** Nach jedem Kapitel 5 Selbsttestfragen aus 00_PROJECT beantworten!

---

### 5.3 Quick-Reference für Corvin

**Was wurde aktualisiert?**
- ✅ Alle 8 Zusammenfassungen auf Vollständigkeit geprüft (100%)
- ✅ Alle 32 Klausurfragen zugeordnet (100% Abdeckung)
- ✅ Lernplan auf 7 Tage aktualisiert (17.03. – 24.03.)
- ✅ Wichtige Formeln je Zusammenfassung extrahiert
- ✅ Klausurrelevanz pro Thema markiert (🔴/🟡/🟢)

**Struktur der neuen PROJECT-Datei:**
1. Projekt-Status (alle Dateien, Umfang, Abdeckung)
2. Inhaltsverzeichnis aller Zusammenfassungen (Hauptthemen + Formeln + Relevanz)
3. Klausur-Fragen-Abdeckung (32 Fragen → Dateien)
4. Lernplan-Integration (7 Tage Prioritäten)
5. Verfügbare Ressourcen (Dateiliste + Lesereihenfolge)

**Empfohlene Nutzung:**
- **Morgens:** 00_PROJECT überblicken (5 Min) → Tagesfokus wählen
- **Nachmittags:** Selbsttestfragen aus Abschnitt 3 beantworten
- **Abends:** Lernplan-Integration (Abschnitt 4) für nächsten Tag prüfen
- **Tag 7:** Abschnitt 5.2 (Lesereihenfolge) für Wiederholung nutzen

---

## 6. NÄCHSTE SCHRITTE

### Für Corvin (End User):
1. **Heute (17.03.):** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md lesen (Tag 1)
2. **Selbsttest:** 5 Transformers-Fragen aus Abschnitt 3 beantworten
3. **Morgen (18.03.):** 04_ZUSAMMENFASSUNG-02 lesen (Tag 2)

### Für Mona (Main Agent):
- PROJECT-Datei ist aktualisiert und vollständig
- Alle Zusammenfassungen sind verlinkt und geprüft
- Lernplan ist synchron mit LERNPLAN-7-Tage-17-03-bis-24-03.md
- Nächste Prüfung: Tag 3 (19.03.) – RL-Teil-1 Fokus

---

**Letzte Aktualisierung:** 17.03.2026, 16:42 Uhr  
**Erstellt von:** Sub-Agent David (Task: 00_PROJECT-Advanced_ML.md aktualisieren)  
**Status:** ✅ ABGESCHLOSSEN
