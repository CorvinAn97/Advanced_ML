# CHEAT SHEET - Advanced ML Klausur
**DIN A4 beidseitig - Alles was du brauchst**

---

## SEITE 1: TRANSFORMERS & SEQUENCES

### Self-Attention (Muss auswendig!)
```
Attention(Q,K,V) = softmax(QK^T/√d_k)·V

Q = X·W_Q, K = X·W_K, V = X·W_V  (X ∈ ℝ^(n×d_model))
MultiHead = Concat(head_1,...,head_h)·W_O
head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)

d_k = d_model/h  (z.B. 512/8 = 64)
Komplexität: O(n²·d)  (Speicher O(n²)!)
```

**Warum √d_k?** Verhindert Softmax-Sättigung bei großen d_k

### Encoder vs Decoder
|  | Encoder (BERT) | Decoder (GPT) |
|--|---------------|-----------------|
| Attention | Bidirektional | Causal (Mask) |
| Mask | Nein | Ja (obere Dreiecksmatrix) |
| Nutzen | Verstehen | Generieren |

### Positional Encoding
```
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(...)
```

---

## WORD EMBEDDINGS

### TF-IDF (Klausurrelevant!)
```
TF-IDF(w,d,D) = TF(w,d) × log(|D|/|{d∈D:w∈d}|)

Cosine Similarity: cos(v₁,v₂) = (v₁·v₂)/(|v₁||v₂|)
```

### Word2Vec
- **CBOW:** Kontext → Zielwort (schneller, häufige Wörter)
- **Skip-gram:** Zielwort → Kontext (besser für seltene Wörter)
- **Negative Sampling:** Statt softmax über |V|, nur k negative Beispiele

### BPE Algorithmus
```
1. Start: Alle Characters als Vokabular
2. Zähle Paare, finde häufigstes
3. Füge neues Symbol hinzu, ersetze alle Vorkommen
4. Wiederhole bis Ziel-Vokabulargröße
```

---

## RNN & LSTM (Sehr wichtig!)

### Vanilla RNN
```
h_t = tanh(W_h·h_{t-1} + W_x·x_t + b_h)
Parameter: N = d_h·(d_x + d_h + 1)
```

### LSTM - ALLE 6 GLEICHUNGEN (Auswendig!)
```
Forget Gate:   f_t = σ(W_f·[h_{t-1},x_t] + b_f)
Input Gate:    i_t = σ(W_i·[h_{t-1},x_t] + b_i)
Cell Candidate: C̃_t = tanh(W_C·[h_{t-1},x_t] + b_C)
Cell State:    C_t = f_t⊙C_{t-1} + i_t⊙C̃_t
Output Gate:   o_t = σ(W_o·[h_{t-1},x_t] + b_o)
Hidden State:  h_t = o_t⊙tanh(C_t)

Parameter: N = 4·d_h·(d_x + d_h + 1)  (4× mehr als RNN!)
```

**Warum kein Vanishing Gradient?** Cell State: C_t = ... + ... (additiv, nicht multiplikativ mit W_h)

### GRU (2 Gates statt 3)
```
z_t = σ(W_z·[h_{t-1},x_t])      # Update Gate
r_t = σ(W_r·[h_{t-1},x_t])      # Reset Gate
h̃_t = tanh(W·[r_t⊙h_{t-1},x_t]) # Candidate
h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t
```

---

## SEQ2SEQ & ATTENTION

### Bottleneck-Problem
- **Problem:** c = h_T muss ganzen Satz kodieren
- **Lösung:** Attention gibt Zugriff auf alle Encoder-States

### Bahdanau Attention
```
Score:      e_{t,i} = v_a^T·tanh(W_s·s_{t-1} + W_h·h_i)
Gewichte:   α_{t,i} = softmax(e_{t,i})
Context:    c_t = Σ_i α_{t,i}·h_i   (WICHTIG: c_t ändert sich pro t!)
```

### Beam Search
- Behält k beste partielle Sequenzen
- P(Sequenz) = Produkt der bedingten Wahrscheinlichkeiten

---

## SEITE 2: REINFORCEMENT LEARNING

### Return & Discount
```
G_t = Σ_{k=0}^∞ γ^k·r_{t+k+1}

γ ≈ 0: Kurzsichtig    γ ≈ 1: Langsichtig
```

### Value Functions
```
v_π(s) = E_π[G_t|S_t=s]           # State-Value
q_π(s,a) = E_π[G_t|S_t=s,A_t=a]   # Action-Value

v_π(s) = Σ_a π(a|s)·q_π(s,a)
```

### Bellman-Gleichungen (Muss!)
```
Expectation:  v_π(s) = Σ_{s',r} p(s',r|s,a)·[r + γ·v_π(s')]
Optimality:   q*(s,a) = Σ_{s',r} p(s',r|s,a)·[r + γ·max_a' q*(s',a')]
```

### Q-Learning (Off-Policy) - AUSWENDIG!
```
Q(s,a) ← Q(s,a) + α·[r + γ·max_a' Q(s',a') - Q(s,a)]

TD Target:    r + γ·max_a' Q(s',a')
TD Error:      δ = Target - Q(s,a)
```

### SARSA (On-Policy) - Unterschied beachten!
```
Q(s,a) ← Q(s,a) + α·[r + γ·Q(s',a') - Q(s,a)]
                ^^^^ nächste Aktion aus Policy, nicht max!
```

### DQN & Double DQN
```
DQN Loss: L(w) = E[(r + γ·max_a' Q_target(s',a';w⁻) - Q(s,a;w))²]

Double DQN (Overestimation fix):
1. a* = argmax_a' Q(s',a';w)           # Selektion mit Q-Network
2. Target = r + γ·Q_target(s',a*;w⁻)   # Bewertung mit Target Network
```

**Target Networks:** Periodisch aktualisiert (alle C Schritte), verhindert "Moving Target"

**Experience Replay:**
- Bricht Korrelationen (i.i.d. Samples)
- Höhere Sample-Effizienz
- Glatteres Training

---

## EXPLORATION

### ε-Greedy
```
π(a|s) = 1-ε+ε/|A|  wenn a = argmax Q(s,a')
         ε/|A|      sonst
```

### UCB (Upper Confidence Bound) - FORMEL!
```
a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]
          ↑           ↑
       Schätzung   Exploration Bonus
```

### Thompson Sampling
```
1. Sample θ ~ Posterior(θ|Data)
2. Wähle a = argmax_a E[r|a,θ]
3. Update Posterior nach Beobachtung
```

---

## GANS & VAES

### GAN Loss (Minimax)
```
L_D = -[log(D(x)) + log(1-D(G(z)))]
L_G = -log(D(G(z)))   # oder max log(D(G(z)))
```

**Mode Collapse:** Generator produziert nur eine Mode
→ Lösung: WGAN (Wasserstein Loss), Minibatch Discrimination

### VAE - ELBO & Reparametrisierung
```
ELBO: L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
           ↑                ↑
      Reconstruction    Regularization

Reparametrisierungs-Trick: z = μ(x) + σ(x)⊙ε, ε~N(0,I)
KL-Divergenz (Gauss): D_KL = 0.5·Σ(μ² + σ² - log(σ²) - 1)
```

---

## DIFFUSION

### Forward Process (Closed Form)
```
x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε,   ε~N(0,I)

ᾱ_t = Π_{s=1}^t α_s   (kumuliert)
```

### Training
```
L = ||ε - ε_θ(x_t, t)||²
gelernt: Rauschen-Vorhersage (nicht x_0 direkt!)
```

### Classifier-Free Guidance (CFG)
```
ε̂ = ε_θ(x_t,t,∅) + w·(ε_θ(x_t,t,c) - ε_θ(x_t,t,∅))
    ↑                ↑
 unbedingt      Konditionierung

w = Guidance Scale (w=1: normal, w>1: stärkere Einhaltung)
```

---

## XAI (Explainable AI)

### LIME
- Lokale Approximation mit interpretablem Modell
- Perturbation der Eingabe, Gewichtung nach Nähe

### SHAP
```
φ_i = Σ_{S⊆N\\{i}} (|S|!(|N|-|S|-1)!/|N|!) · [v(S∪{i}) - v(S)]

Eigenschaften: Additiv, Konsistent, Fair
```

### Counterfactuals
- "Was wäre, wenn..." minimale Änderung → gewünschte Vorhersage

---

## IMITATION LEARNING

### Behavioral Cloning
```
L = -E[log π_θ(a|s)]   # Supervised Learning auf Demo-Daten

PROBLEM: Distributional Shift
Fehler akkumulieren: O(εT²) statt O(εT)
```

### DAgger (Dataset Aggregation)
```
1. Starte mit π_1 = Behavior Cloning
2. Führe π_i aus, sammle Zustände
3. Label mit Experten
4. Trainiere π_{i+1} auf allen Daten
5. Wiederhole

Garantie: O(εT) (linear statt quadratisch!)
```

---

## RLHF & DPO

### RLHF Pipeline
```
1. SFT: Supervised Fine-Tuning auf Demo-Daten
2. Reward Modeling: Trainiere R(x,y) aus Präferenzen (paarweise Vergleiche)
3. PPO: Policy Gradient mit KL-Penalty zum SFT-Modell
```

### DPO (Direct Preference Optimization)
```
L_DPO = -E[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]

→ Kein Reward-Modell nötig!
→ Einfacher als RLHF
→ Aber: π_ref muss gut sein
```

---

## OFFLINE RL

### Problem: Distribution Shift
- Policy lernt aus festem Datensatz D
- Kann keine neuen Aktionen ausprobieren
- OOD (Out-of-Distribution) Aktionen: Q-Werte extrapolieren → instabil

### CQL (Conservative Q-Learning)
```
L_CQL = α·E_{s,a~μ}[Q(s,a)] - α·E_{(s,a)~D}[Q(s,a)] + L_TD(Q)
        ↑                           ↑
   Vermindere für              Erhöhe für
   OOD-Aktionen               in-Distribution
```

### IQL (Implicit Q-Learning)
- Keine expliziten OOD-Aktionen nötig
- Expectile Loss statt max
- Asymmetrischer Loss: τ|x|² wenn x>0, (1-τ)|x|² wenn x<0

---

## WICHTIGSTE FORMELN (Priorität 🔴)

1. **Self-Attention:** `softmax(QK^T/√d_k)·V`
2. **LSTM:** 6 Gleichungen (oben)
3. **Q-Learning:** `Q ← Q + α[r + γ·max Q' - Q]`
4. **Bellman-Optimality:** `q*(s,a) = E[r + γ·max q*(s',a')]`
5. **UCB:** `argmax[μ̂ + √(2ln(T)/N)]`
6. **ELBO:** `E[log p(x|z)] - KL(q(z|x)||p(z))`
7. **KL-Divergenz:** `0.5·Σ(μ² + σ² - log(σ²) - 1)`
8. **VAE Reparametrisierung:** `z = μ + σ⊙ε`
9. **Diffusion Forward:** `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`
10. **CFG:** `ε̂ = ε_∅ + w·(ε_c - ε_∅)`

---

## NOTFALL-MERKHILFEN

**Transformer vs RNN:**
- Transformer: O(1) Pfadlänge, O(n²) Speicher
- RNN: O(n) Pfadlänge, O(n) Speicher

**Q-Learning vs SARSA:**
- Q-Learning: max über nächste Aktion (Off-Policy)
- SARSA: tatsächliche nächste Aktion (On-Policy)

**GAN vs VAE:**
- GAN: Adversarial, kann Mode Collapse haben
- VAE: Probabilistisch, ELBO, Latent Space strukturiert
- Diffusion: Langsam aber stabil, kein Mode Collapse

**Bandits:**
- UCB: Deterministisch, theoretische Garantien
- Thompson Sampling: Probabilistisch, oft besser in Praxis

---

**Viel Erfolg! 🎯**
