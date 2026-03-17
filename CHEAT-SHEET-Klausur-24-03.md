# 🎯 Advanced ML Cheat Sheet – Klausur 24.03.2026
**Doppelseitig DIN A4 | Priorisiert nach Klausurrelevanz | Aktualisiert mit PDF-Inhalten**

---

## 📄 SEITE 1: THEORIE & KONZEPTE

### 🔢 DIE 15 WICHTIGSTEN FORMELN (Erweitert)

| # | Formel | Thema | Bedeutung | PDF-Quelle |
|---|--------|-------|-----------|------------|
| 1 | `Attention(Q,K,V) = softmax(QK^T/√d_k)·V` | Self-Attention | Kern des Transformers | 03-Transformers |
| 2 | `Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]` | Q-Learning | Off-Policy Update | 07-RL-Teil-1 |
| 3 | `f_t = σ(W_f·[h_{t-1},x_t] + b_f)` | LSTM Forget Gate | Welche Info löschen? | 02-WordEmbeddings |
| 4 | `C_t = f_t⊙C_{t-1} + i_t⊙C̃_t` | LSTM Cell State | Gedächtnis-Update | 02-WordEmbeddings |
| 5 | `G_t = Σ γ^k·r_{t+k+1}` | Return | Diskontierter kumulativer Reward | 07-RL-Teil-1 |
| 6 | `L = -[log(D(x)) + log(1-D(G(z)))]` | GAN Loss | Generator vs Discriminator | 05-GANs-VAEs |
| 7 | `x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε` | Diffusion Forward | Noise Addition (closed form) | 05-GANs-VAEs |
| 8 | `L = E[||x-x̂||²] + D_KL(q(z|x)\|\|p(z))` | VAE ELBO | Reconstruction + KL-Regularization | 05-GANs-VAEs |
| 9 | `a = argmax_a[μ̂_a + √(2·ln(T)/N(a))]` | UCB | Exploration: Optimismus bei Unsicherheit | 08-RL-Teil-2 |
| 10 | `W' = W + B·A` (r << d) | LoRA | Low-Rank Adaptation für PEFT | 03-Transformers |
| 11 | `v_π(s) = E_π[R_{t+1} + γv_π(S_{t+1}) | S_t = s]` | Bellman Equation | Rekursive Wertfunktion | 07-RL-Teil-1 |
| 12 | `tf(w,d) = #w in d / #Wörter in d` | TF-IDF | Term Frequency | 02-WordEmbeddings |
| 13 | `idf(w,D) = log(|D| / #docs containing w)` | TF-IDF | Inverse Document Frequency | 02-WordEmbeddings |
| 14 | `φ_i = Σ_{S⊆N\i} (|S|!(|N|-|S|-1)!/|N|!)·[v(S∪i)-v(S)]` | SHAP | Shapley Value Formel | 04-XAI |
| 15 | `L_W = E[c(x)] - E[c(G(z))] + λE[(||∇c(x̂)||-1)²]` | WGAN-GP | Wasserstein Loss + Gradient Penalty | 05-GANs-VAEs |

---

### 🔑 SCHLÜSSELKONZEPTE KOMPAKT (Erweitert)

#### **Word2Vec & Embeddings**
- **CBOW:** Kontext → Zielwort (schneller, mittelt Kontext)
- **Skip-gram:** Zielwort → Kontext (besser für seltene Wörter)
- **Distributional Hypothesis:** "Wort bekannt durch Kontext" (Firth 1957)
- **FastText:** Subword n-grams (3-6) → behandelt OOV, bessere seltene Wörter
- **BPE:** Iterative Fusion häufigester Zeichenfolgen, kontrollierte Vokabulargröße
- **Bias in Embeddings:** Verstärken gesellschaftliche Vorurteile (Caliskan et al. 2017)

#### **LSTM Gates (3 Stück) + Varianten**
| Gate | Formel | Funktion |
|------|--------|----------|
| **Forget** | `f_t = σ(W_f·[h_{t-1},x_t])` | Löschen aus Cell State (0=löschen, 1=behalten) |
| **Input** | `i_t = σ(W_i·[h_{t-1},x_t])` | Neue Info hinzufügen |
| **Output** | `o_t = σ(W_o·[h_{t-1},x_t])` | Filtern des Outputs |
- **Peephole Connections:** Gates hängen auch von Cell State ab
- **Coupled Gates:** Forget + Input gekoppelt (was vergessen wird, wird ersetzt)
- **Parameter Count:** `params = 4 × [(x + h) × h + h]` pro LSTM-Layer

#### **Transformer-Architektur (Erweitert)**
- **Encoder:** Bidirektional, Self-Attention nur, für Understanding
- **Decoder:** Kausal (masked), Self-Attention + Cross-Attention (K,V von Encoder, Q von Decoder)
- **Multi-Head:** h=8-16 parallele Attention-Heads (verschiedene Aspekte), concat + W^O
- **LayerNorm:** Pre-Norm (modern, stabiler) vs Post-Norm (original)
- **FFN:** SwiGLU (SOTA: Swish(xW+b)⊗xV+c, d_ff≈2/3×4d) oder GELU/ReLU (d_ff=4d)
- **Positional Encoding:** Sinus/Cosinus (original) vs RoPE (modern, bessere Extrapolation)
- **Komplexität:** O(n²·d) durch Self-Attention, Context Window limitiert

---

### ⚖️ VERGLEICHE (Erweitert)

#### **RNN vs LSTM vs Transformer**
| Aspekt | RNN | LSTM | Transformer |
|--------|-----|------|-------------|
| **Parallelisierung** | ❌ sequentiell | ❌ sequentiell | ✅ vollständig |
| **Lange Abhängigkeiten** | ❌ Vanishing Gradient | ✅ Gut (Cell State) | ✅ Sehr gut (O(1) Pfad) |
| **Komplexität/Token** | O(d²) | O(4·d²) | O(n²·d) |
| **State** | Hidden State | Cell + Hidden | Keine Rekurrenz |
| **Status** | Veraltet | Legacy | SOTA |

#### **GAN vs VAE vs Diffusion**
| Aspekt | GAN | VAE | Diffusion |
|--------|-----|-----|-----------|
| **Bildqualität** | ✅ Scharf | ❌ Verwaschen | ✅✅ Sehr scharf |
| **Training** | ❌ Instabil (Mode Collapse) | ✅ Stabil | ✅ Stabil |
| **Mode Collapse** | ❌ Ja (WGAN löst) | ✅ Nein | ✅ Nein |
| **Sampling-Speed** | ✅ Schnell (1 Schritt) | ✅ Schnell | ❌ Langsam (100-1000 Schritte) |
| **Latent Space** | ❌ Unstrukturiert | ✅ Strukturiert (KL) | ✅ Strukturiert |
| **SOTA?** | ❌ Nein | ❌ Nein | ✅✅ Ja (Stable Diffusion) |

#### **XAI-Methoden Vergleich**
| Methode | Typ | Lokal/Global | Modellunabh. | Rechenintensiv | Formel |
|---------|-----|--------------|--------------|----------------|--------|
| **LIME** | Post-hoc | Lokal | ✅ Ja | ✅ Mittel | Surrogate Model |
| **SHAP** | Post-hoc | Lokal+Global | ✅ Ja | ✅✅ Hoch | Shapley Values |
| **PFI** | Post-hoc | Global | ✅ Ja | ✅✅ Hoch | ΔAccuracy nach Permutation |
| **Lineare Regression** | Intrinsisch | Global | ❌ Nein | ✅ Niedrig | y = w₀ + w₁x₁ + ... |
| **Entscheidungsbaum** | Intrinsisch | Global | ❌ Nein | ✅ Niedrig | Feature Importance = ΣΔimpurity |

---

### ❓ TYPISCHE KLAUSURFRAGEN (25+ Stück, erweitert)

#### **Transformers (🔴 Sehr wichtig)**
1. **Q:** Was ist Self-Attention? Erklären Sie Q, K, V!
   **A:** Query, Key, Value aus Input berechnet (q=W_Q·x, k=W_K·x, v=W_V·x). Attention = softmax(QK^T/√d_k)·V. Q sucht relevante Keys, V enthält Information.

2. **Q:** Warum braucht man Positional Encoding?
   **A:** Attention ist positionsunabhängig (permutationsinvariant). PE (sinus/cosinus oder RoPE) addiert Positionsinfo zu Embeddings.

3. **Q:** Unterschied Encoder vs Decoder?
   **A:** Encoder: bidirektional, nur Self-Attention. Decoder: kausal (masked), Self-Attention + Cross-Attention (K,V von Encoder).

4. **Q:** Was ist Multi-Head Attention?
   **A:** Mehrere Attention-Heads parallel (h=8-16). Jeder Head lernt andere Beziehungen (Q_i=W_Q^i·x, etc.). Concat + linear projection W^O.

5. **Q:** Was ist Masked Attention?
   **A:** Im Decoder: zukünftige Tokens auf -∞ setzen vor Softmax. Verhindert "Cheating" bei Generierung (autoregressiv).

6. **Q:** Warum Transformer besser als RNN für lange Sequenzen?
   **A:** O(1) Pfadlänge zwischen beliebigen Wörtern (vs O(n) bei RNN), vollständige Parallelisierung, kein Vanishing Gradient durch Rekurrenz.

7. **Q:** SwiGLU vs GELU im FFN?
   **A:** SwiGLU: Swish(xW+b)⊗xV+c (gated, 2 Projektionen), reduziert d_ff auf ~2/3×4d bei gleicher Parameterzahl. Empirisch besser für LLMs.

#### **LSTM & RNNs (🔴 Sehr wichtig)**
8. **Q:** Erklären Sie die 3 LSTM Gates!
   **A:** Forget (löschen: f_t⊙C_{t-1}), Input (hinzufügen: i_t⊙C̃_t), Output (filtern: o_t⊙tanh(C_t)). Cell State = f⊙C_prev + i⊙C_candidate.

9. **Q:** Vanishing Gradient Problem?
   **A:** Gradienten werden bei Backprop durch viele Schichten exponentiell klein (∂L/∂W ∝ Π∂h_t/∂h_{t-1}). LSTM löst durch expliziten Cell State (Gradient Flow).

10. **Q:** Unterschied LSTM vs GRU?
    **A:** GRU: 2 Gates (Update, Reset), kein Cell State, weniger Parameter (3× vs 4×), schneller. Update Gate ersetzt Forget+Input.

11. **Q:** Bidirectional LSTM?
    **A:** Zwei LSTMs: vorwärts + rückwärts, Outputs konkateniert. Kennt linken + rechten Kontext. Für Sequence Labeling (NER, POS).

#### **Word Embeddings (🔴 Sehr wichtig)**
12. **Q:** CBOW vs Skip-gram?
    **A:** CBOW: Kontext→Ziel (schneller, mittelt). Skip-gram: Ziel→Kontext (besser für seltene Wörter, langsamer).

13. **Q:** Distributional Hypothesis?
    **A:** "Wort bekannt durch Kontext" (Firth 1957). Ähnliche Kontexte → ähnliche Bedeutung → ähnliche Vektoren.

14. **Q:** FastText vs Word2Vec?
    **A:** FastText: Subword n-grams (3-6 Buchstaben), Summe der n-gram Vektoren. Löst OOV, besser für Morphologie, seltenere Wörter.

15. **Q:** TF-IDF Formel?
    **A:** tf(w,d) = #w in d / #Wörter in d (lokale Wichtigkeit). idf(w,D) = log(|D| / #docs mit w) (globale Seltenheit). TF-IDF = tf × idf.

16. **Q:** BPE (Byte Pair Encoding)?
    **A:** Iterative Fusion häufigester Zeichenfolgen. Start: alle Zeichen. Repeat: füge häufigstes N-Gramm hinzu bis Vokabulargröße erreicht. Kompromiss Word/Character-Level.

#### **RL (🔴 Sehr wichtig)**
17. **Q:** Q-Learning Update-Formel?
    **A:** Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]. Off-Policy, lernt optimale Policy. TD-Target: r + γ·max_a'Q(s',a').

18. **Q:** Unterschied Q-Learning vs SARSA?
    **A:** Q-Learning: max_a' (off-policy, optimistisch). SARSA: tatsächliche nächste Aktion a' (on-policy, vorsichtiger). SARSA: Q(s,a) ← Q + α[r + γ·Q(s',a') - Q].

19. **Q:** Overestimation Problem bei DQN?
    **A:** max Operator überschätzt Q-Werte (E[max X₁,X₂] ≥ max(E[X₁],E[X₂])). Double DQN löst: Selektion mit Q, Bewertung mit Target-Netzwerk.

20. **Q:** Wozu Target Networks?
    **A:** Stabilisiert Training. Target w⁻ wird periodisch kopiert (oder Polyak: w⁻ ← τw⁻ + (1-τ)w), verhindert "Moving Target".

21. **Q:** UCB Formel?
    **A:** a = argmax_a[μ̂_a + √(2·ln(T)/N(a))]. Optimismus bei Unsicherheit. Bonus ↓ mit N(a), √(2·ln(T)) ↑ mit Zeit.

22. **Q:** Thompson Sampling?
    **A:** Sample θ ~ p(θ̂) (belief state), wähle optimale Aktion unter Sample. Posterior Sampling. Gut empirisch, schwer zu analysieren.

23. **Q:** Bellman Equation?
    **A:** v_π(s) = E_π[R_{t+1} + γv_π(S_{t+1}) | S_t = s]. Rekursiv. Optimal: v*(s) = max_a E[R + γv*(S')].

#### **Generative Modelle (🔴 Sehr wichtig)**
24. **Q:** Mode Collapse bei GANs?
    **A:** Generator produziert nur 1 Mode (z.B. nur "6" bei MNIST). Ursache: D zu gut, Gradient verschwindet. WGAN mit Gradient Penalty löst.

25. **Q:** VAE Reparametrisierungs-Trick?
    **A:** z = μ + σ⊙ε, ε~N(0,I). Macht z differenzierbar für Backprop (Gradient durch Sampling).

26. **Q:** Diffusion Forward/Reverse Process?
    **A:** Forward: q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I) (Markov, Noise). Reverse: p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ).

27. **Q:** Classifier-Free Guidance?
    **A:** ε̂ = ε_θ(∅) + w·(ε_θ(c) - ε_θ(∅)). Kein separater Classifier nötig. w=1: unconditional, w>1: stronger guidance.

28. **Q:** Wasserstein Loss vs BCE?
    **A:** BCE: vanishing Gradient bei perfekten D. WGAN: L = E[c(x)] - E[c(G(z))] mit 1-Lipschitz (||∇c||≤1). Misst Earth Mover's Distance.

#### **XAI (🟡 Wichtig)**
29. **Q:** LIME vs SHAP?
    **A:** LIME: lokale lineare Approximation (Surrogate Model, pertubieren). SHAP: Shapley Values (fairen Beitrag, theoretisch fundiert, additiv).

30. **Q:** SHAP Formel?
    **A:** φ_i = Σ_{S⊆N\i} (|S|!(|N|-|S|-1)!/|N|!)·[v(S∪i)-v(S)]. Fairer Beitrag jedes Features.

31. **Q:** PFI Problem bei korrelierten Features?
    **A:** Korrelierte Features teilen Wichtigkeit (Permutation eines hat kleinen Effekt). Lösung: Clustere Features, behalte eines pro Cluster.

#### **Imitation Learning (🟡 Wichtig)**
32. **Q:** Distributional Shift bei Behavioral Cloning?
    **A:** p_test(o_t) ≠ p_train(o_t). Fehler bringen Agent in Zustände nicht in Trainingsdaten. Folge: O(εT²) Fehler (vs O(εT) i.i.d.).

33. **Q:** DAgger Algorithmus?
    **A:** 1. Train π_0 auf Experten-Daten D. 2. Rollout π_i, sammle D_π. 3. Experte labelt D_π. 4. D ← D ∪ D_π. 5. π_{i+1} = train(D). Löst Distributional Shift (O(T) statt O(T²)).

---

### 🎓 DOZENTEN-HINWEISE (aus PDFs markiert)

| Thema | Hinweis | PDF |
|-------|---------|-----|
| **Transformer** | "Attention is not all you need" – Positional Encoding kritisch! | 03-Transformers |
| **LSTM** | Peephole Connections und Coupled Gates sind bekannte Varianten | 02-WordEmbeddings |
| **RL** | GLIE (Greedy in Limit with Infinite Exploration) garantiert Konvergenz | 07-RL-Teil-1 |
| **GAN** | Discriminator und Generator sollten vergleichbar stark sein | 05-GANs-VAEs |
| **XAI** | Intrinsisch interpretierbare Modelle (lineare Regression, Bäume) oft ausreichend | 04-XAI |
| **IL** | Causal Confusion: Policy lernt falsche Kausalitäten (de Haan et al.) | 06-IL |
| **Embeddings** | Bias in Embeddings verstärkt gesellschaftliche Vorurteile | 02-WordEmbeddings |

---

## 📄 SEITE 2: ALGORITHMEN & ANWENDUNG

### 🔄 ALGORITHMUS-SCHRITTE (Präzisiert)

#### **Backpropagation Through Time (BPTT)**
```
1. Forward Pass: Berechne alle h_t, y_t (sequentiell!)
2. Loss: L = Σ_t l(y_t, ŷ_t)
3. Backward: δ_t = ∂L/∂z_t vom Output zurück propagieren
   - δ_t = (W^T · δ_{t+1}) ⊙ g'(z_t) für RNN
4. Gradient: ∂L/∂W = Σ_t δ_t · h_{t-1}
5. Update: W ← W - α·∂L/∂W
```
⚠️ **Truncated BPTT:** Bei langen Sequenzen begrenzen (z.B. 50 Schritte)
⚠️ **Vanishing Gradient:** δ_t ∝ Π_{k=t}^T g'(z_k) → 0 bei T>>t

#### **Training Loop (DQN) – Erweitert**
```
1. Initialize Q(w), Target Q(w⁻), Replay Buffer D
2. For each episode:
   - For each step:
     * Choose action: a = argmax_a Q(s,a) mit ε-greedy
     * Execute, observe (s,a,r,s')
     * Store in D
   - Sample minibatch (s,a,r,s') ~ uniform(D)
   - Compute target: y = r + γ·max_a' Q_target(s',a'; w⁻)
   - Gradient descent: w ← w - α·∇_w L(y, Q(s,a;w))
   - Periodically: w⁻ ← w (oder Polyak: w⁻ ← τw⁻ + (1-τ)w)
```
✅ **Experience Replay:** Unkorrelierte Daten, höhere Effizienz
✅ **Target Network:** Verhindert Moving Target

#### **Policy Gradient (REINFORCE)**
```
1. Sample Episode: τ = (s_0,a_0,r_0,...,s_T) unter π_θ
2. Compute Return: G_t = Σ_{k=0}^{T-t} γ^k·r_{t+k+1}
3. Update: θ ← θ + α·Σ_t ∇_θ log π_θ(a_t|s_t)·G_t
```
✅ **Vorteil:** Konvergiert zu lokalem Optimum, on-policy
❌ **Nachteil:** Hohe Varianz (Monte-Carlo), slow

#### **DAgger (Imitation Learning) – Distributional Shift Lösung**
```
1. Train π_0 auf Experten-Daten D = {(o_0,a*_0), ...}
2. Repeat K times:
   - Rollout π_i, sammle D_π = {(o_0,...,o_T)}
   - Experte labelt D_π mit korrekten Aktionen a*_t
   - D ← D ∪ D_π (Dataset Aggregation)
   - π_{i+1} = train(D)
```
✅ **Löst:** Distributional Shift Problem (O(T) statt O(T²))
❌ **Nachteil:** Experte muss online labelen (aufwendig)

#### **Wasserstein GAN Training**
```
1. Initialize Generator G, Critic c (nicht D!)
2. For each iteration:
   - Train Critic (n_critic Schritte):
     * Sample real x ~ p_data, noise z ~ p(z)
     * Sample x̂ = εx + (1-ε)G(z) (Interpolation)
     * Loss: L = E[c(x)] - E[c(G(z))] + λE[(||∇_x̂ c(x̂)||-1)²]
     * Gradient descent auf c
   - Train Generator:
     * Sample z ~ p(z)
     * Loss: L_G = -E[c(G(z))]
     * Gradient descent auf G
```
✅ **Vorteil:** Kein Vanishing Gradient, stabil
✅ **Gradient Penalty:** Erzwingt 1-Lipschitz (||∇c||≤1)

---

### 🎛️ HYPERPARAMETER & TRICKS (Erweitert)

#### **Learning Rate Scheduling**
| Methode | Formel | Use Case |
|---------|--------|----------|
| **Step Decay** | α·γ^k nach k Epochen | Standard |
| **Exponential** | α·γ^t | Kontinuierlich |
| **Cosine Annealing** | α·0.5·(1+cos(π·t/T)) | SOTA (Transformer) |
| **Warmup** | Linear ↑ dann ↓ | Large Models (Transformer, LLM) |

#### **Dropout**
- **Training:** Zufällig Neuronen auf 0 setzen (p=0.1-0.5)
- **Inferenz:** Alle Neuronen aktiv, Output skalieren mit (1-p)
- **Effekt:** Verhindert Co-Adaptation, reduziert Overfitting

#### **Layer Normalization**
```
LN(x) = γ⊙(x-μ)/√(σ²+ε) + β
- μ = E[x], σ² = Var[x] über Feature-Dimensionen
- γ, β: lernbare Parameter (init: γ=1, β=0)
- Pro Datenpunkt (nicht Batch!)
- Pre-Norm: Vor Sub-Layer (stabiler, modern)
```

#### **Gradient Clipping**
```
if ||g|| > clip: g = g · clip/||g||
- Typisch: clip=1.0
- Verhindert Exploding Gradients (RNNs!)
```

#### **Batch Size Trade-offs**
- **Klein (32-128):** Mehr Rauschen, bessere Generalization
- **Groß (256-1024):** Schneller, stabiler, schlechtere Generalization
- **GPU Memory:** Limitiert Batch Size

---

### 🤖 RL-ALGORITHMEN ÜBERSICHT (Erweitert)

| Algorithmus | On/Off-Policy | Value/Policy | Key Feature | Use Case |
|-------------|---------------|--------------|-------------|----------|
| **Q-Learning** | ❌ Off | Value | max_a' Q(s',a') | Tabular, kleine Spaces |
| **SARSA** | ✅ On | Value | Q(s',a') tatsächlich | Safety-critical |
| **DQN** | ❌ Off | Value | Experience Replay, Target Net | Pixel Input (Atari) |
| **Double DQN** | ❌ Off | Value | Entkoppelt Selektion/Bewertung | Overestimation Fix |
| **REINFORCE** | ✅ On | Policy | Monte-Carlo Gradient | Small Actions |
| **Actor-Critic** | ✅ On | Hybrid | Value schätzt Advantage A=Q-V | Continuous Actions |
| **UCB** | - | Bandit | √(2·ln(T)/N(a)) Bonus | Exploration (Bandits) |
| **Thompson** | - | Bandit | Sample θ ~ p(θ̂) | Exploration (empirisch gut) |

#### **Double DQN – Overestimation Fix**
```
1. Selektion: a* = argmax_a Q(s',a; w)  (mit Q-Netzwerk)
2. Bewertung: y = r + γ·Q_target(s', a*; w⁻)  (mit Target-Netzwerk)
```
✅ **Dekorrelation:** Rauschen zwischen Selektion und Bewertung entkoppelt

#### **Actor-Critic**
- **Actor:** Policy π_θ(a|s) → Update mit Policy Gradient
- **Critic:** Value V_w(s) → schätzt Advantage A(s,a) = Q(s,a) - V(s)
- **Vorteil:** Niedrigere Varianz als REINFORCE (Bootstrapping)

---

### ⚠️ EDGE CASES & FALLSTRICKE (Erweitert)

#### **Vanishing/Exploding Gradients**
- **Problem:** Gradienten → 0 oder → ∞ bei tiefen Netzen / langen Sequenzen
- **Symptome:** Training stagniert oder divergiert
- **Lösungen:**
  - LSTM/GRU statt Vanilla RNN (Cell State)
  - Gradient Clipping (||g|| > clip: g = g·clip/||g||)
  - Residual Connections (a_{l+1} = F(a_l) + a_l)
  - Layer Normalization
  - GELU/ReLU statt Sigmoid/Tanh

#### **Mode Collapse (GANs)**
- **Problem:** Generator produziert nur 1 Mode (z.B. nur "6" bei MNIST)
- **Ursache:** Discriminator zu gut, Gradient verschwindet (Vanishing Gradient)
- **Lösungen:**
  - Wasserstein GAN (Critic statt D, EMD)
  - Gradient Penalty (||∇c(x̂)||-1)²)
  - Mini-Batch Discrimination
  - Experience Replay (Buffer)

#### **Catastrophic Forgetting**
- **Problem:** Fine-tuning zerstört vorheriges Wissen
- **Symptome:** Performance auf alten Tasks sinkt drastisch
- **Lösungen:**
  - **LoRA:** Nur Low-Rank Adapter trainieren (W' = W + B·A)
  - **Elastic Weight Consolidation:** Wichtige Parameter schützen (Fisher Info)
  - **Replay Buffer:** Alte Daten mittrainieren
  - **Prompt Tuning:** Nur Soft Prompts trainieren

#### **Distributional Shift (Imitation Learning)**
- **Problem:** p_train(state) ≠ p_test(state)
- **Folge:** Fehler akkumulieren sich (O(εT²) vs O(εT) i.i.d.)
- **Lösung:** DAgger (iterative Datensammlung mit Policy-Rollouts + Experten-Labeling)

#### **Overestimation Bias (DQN)**
- **Problem:** max Operator überschätzt Q-Werte systematisch (E[max X₁,X₂] ≥ max(E[X₁],E[X₂]))
- **Lösung:** Double DQN (Selektion mit Q, Bewertung mit Target)

#### **Offline RL Distribution Shift**
- **Problem:** Q extrapoliert für ungesehene Aktionen (OOD)
- **Lösungen:**
  - Conservative Q-Learning (Q für OOD Aktionen minimieren)
  - Behavior Regularization (Policy nahe π^D halten)

#### **KL-Divergence bei VAE (Posterior Collapse)**
- **Problem:** Posterior kollabiert zu Prior (μ→0, σ→1)
- **Folge:** Nur Reconstruction Loss, kein strukturierter Latent Space
- **Lösung:** β-VAE (KL stärker gewichten: L = Rec + β·KL)

#### **Causal Confusion (Imitation Learning)**
- **Problem:** Policy lernt falsche Kausalitäten (z.B. Blinklicht → Abbiegen, nicht Umgekehrt)
- **Folge:** Scheitert bei Distribution Shift
- **Lösung:** Multi-Task, Goal-Conditioned BC, oder Attention auf relevante Features

---

### 🔍 EXPLORATION IN DEEP RL (Erweitert)

| Methode | Idee | Formel | Use Case |
|---------|------|--------|----------|
| **ε-greedy** | Zufall mit Wahrscheinlichkeit ε | π(a) = ε/m sonst 1-ε+ε/m | Simple, GLIE nötig |
| **UCB** | Optimismus bei Unsicherheit | a = argmax[μ̂ + √(2ln(T)/N(a))] | Bandits, optimal |
| **Thompson** | Sample from belief | θ ~ p(θ̂), a = argmax E[r|θ] | Empirisch stark |
| **Count-Based** | Bonus für neue Zustände | r̂ = r + B(N(s)) | Large State Spaces |
| **Pseudo-Count** | Dichtemodell für N(s) | N̂(s) aus p(s), p'(s) | Very Large Spaces |
| **Bootstrapped DQN** | Ensemble von Q-Netzen | Sample Q_i, act 1 Episode | Deep Exploration |

---

## 📊 KLAUSUR-STRATEGIE (Erweitert)

✅ **Prioritäten:** Transformers (3-4h), Q-Learning (2.5h), LSTM (2h), GANs/Diffusion (2h) zuerst
✅ **Formeln auswendig:** Attention, Q-Learning, LSTM Gates, UCB, Bellman, SHAP, TF-IDF
✅ **Konzepte verstehen > auswendig lernen:** Warum? Wann? Trade-offs?
✅ **Zeitmanagement:** Einfache Fragen zuerst, Unsichere markieren, am Ende zurück
✅ **Dozenten-Hinweise beachten:** Positional Encoding, GLIE, Mode Collapse, DAgger

---

*Erstellt: 17.03.2026 | Aktualisiert: 17.03.2026 09:45 | Basierend auf 8 PDF-Vorlesungsunterlagen + 00_PROJECT-Advanced_ML.md*
