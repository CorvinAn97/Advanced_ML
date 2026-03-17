# ZUSAMMENFASSUNG 07: Reinforcement Learning Teil 1

## Übersicht
- **Seitenzahl:** ~90 Seiten (AdvancedML-07-RL-Teil-1.pdf)
- **Hauptthemen:** RL-Grundlagen, Value-Based Methods, Q-Learning, DQN, Double DQN
- **Klausurrelevanz:** SEHR HOCH - Q-Learning und DQN gehören zu den wichtigsten Themen

---

## Detaillierte Inhalte

### 1. RL-Grundlagen

#### Kernkonzepte

**Agent-Environment-Interaktion:**
```
State (s_t) → Agent → Action (a_t) → Environment → Reward (r_t), State (s_{t+1})
```

- **Agent:** Entscheidungs-Einheit, die lernt und handelt
- **Environment:** Umgebung, in der der Agent agiert
- **State (s_t):** Zustand des Environments zum Zeitpunkt t
- **Action (a_t):** Aktion, die der Agent zum Zeitpunkt t ausführt
- **Reward (r_t):** Skalares Feedback-Signal vom Environment

#### Policy π

Die Policy definiert das Verhalten des Agenten:

```
a_t = π(s_t)          (deterministisch)
a_t ~ π(·|s_t)        (stochastisch)
```

- **Deterministisch:** Eine bestimmte Aktion für jeden Zustand
- **Stochastisch:** Wahrscheinlichkeitsverteilung über Aktionen

#### Return G_t

Der Return ist der kumulative, diskontierte Reward:

```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ... = Σ_{k=0}^{∞} γ^k · r_{t+k+1}
```

- **γ (Gamma):** Discount-Faktor ∈ [0,1]
  - γ ≈ 0: Kurzfristige Rewards wichtig
  - γ ≈ 1: Langfristige Rewards wichtig
- **Episodische Tasks:** Episode endet bei terminalem Zustand
- **Continuing Tasks:** Keine terminalen Zustände, unendlicher Horizont

#### Reward-Hypothese

> **"Alles, was wir als Ziel betrachten, kann als Maximierung eines erwarteten kumulativen Rewards formalisiert werden."**

Diese Hypothese ist fundamental für RL - jedes Ziel lässt sich als Reward-Funktion ausdrücken.

---

### 2. Value Functions (Wertfunktionen)

Value Functions quantifizieren, wie "gut" es ist, sich in einem bestimmten Zustand zu befinden oder eine bestimmte Aktion auszuführen.

#### State-Value Function v_π(s)

Erwarteter Return, ausgehend von Zustand s unter Policy π:

```
v_π(s) = E_π[G_t | S_t = s]
       = E_π[r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ... | S_t = s]
```

**Rekursive Form (Bellman-Gleichung):**
```
v_π(s) = E_π[r_{t+1} + γ·v_π(S_{t+1}) | S_t = s]
```

#### Action-Value Function q_π(s,a) (Q-Function)

Erwarteter Return bei Ausführung von Aktion a in Zustand s unter Policy π:

```
q_π(s,a) = E_π[G_t | S_t = s, A_t = a]
         = E_π[r_{t+1} + γ·r_{t+2} + ... | S_t = s, A_t = a]
```

**Rekursive Form:**
```
q_π(s,a) = E_π[r_{t+1} + γ·q_π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
```

#### Zusammenhang zwischen v_π und q_π

```
v_π(s) = E_{a~π}[q_π(s,a)] = Σ_a π(a|s) · q_π(s,a)
```

#### Optimal Value Functions

**Optimal State-Value Function:**
```
v*(s) = max_π v_π(s)  für alle s ∈ S
```

**Optimal Action-Value Function:**
```
q*(s,a) = max_π q_π(s,a)  für alle s ∈ S, a ∈ A
```

**Optimale Policy π*:**
```
π*(s) = argmax_a q*(s,a)
```

Wenn wir q* kennen, ist die optimale Policy trivial: immer die Aktion mit dem höchsten q*-Wert wählen.

---

### 3. Bellman-Gleichungen

Die Bellman-Gleichungen beschreiben die rekursive Struktur von Value Functions.

#### Bellman-Expectation-Gleichung für v_π

```
v_π(s) = Σ_a π(a|s) · Σ_{s',r} p(s',r|s,a) · [r + γ·v_π(s')]
```

**Interpretation:**
- Erwarte über alle möglichen Aktionen (gemäß π)
- Erwarte über alle möglichen nächsten Zustände und Rewards
- Reward sofort + diskontierter Wert des nächsten Zustands

#### Bellman-Expectation-Gleichung für q_π

```
q_π(s,a) = Σ_{s',r} p(s',r|s,a) · [r + γ·Σ_{a'} π(a'|s') · q_π(s',a')]
```

#### Bellman-Optimality-Gleichung für v*

```
v*(s) = max_a Σ_{s',r} p(s',r|s,a) · [r + γ·v*(s')]
```

#### Bellman-Optimality-Gleichung für q*

```
q*(s,a) = Σ_{s',r} p(s',r|s,a) · [r + γ·max_{a'} q*(s',a')]
```

**Wichtig:** Die Bellman-Optimality-Gleichung für q* ist die Grundlage für Q-Learning!

#### Backup-Diagramme

- **v_π:** Wurzel (s) → alle Aktionen (π) → alle nächsten Zustände (p)
- **q_π:** Wurzel (s,a) → alle nächsten Zustände (p) → alle nächsten Aktionen (π)
- **v*:** Wurzel (s) → max über Aktionen → alle nächsten Zustände
- **q*:** Wurzel (s,a) → alle nächsten Zustände → max über nächste Aktionen

---

### 4. Exploration vs. Exploitation

Fundamentales Dilemma im Reinforcement Learning:

| **Exploration** | **Exploitation** |
|-----------------|------------------|
| Neue Dinge ausprobieren | Bekannte gute Aktionen nutzen |
| Langfristiger Gewinn | Kurzfristiger Gewinn |
| Risiko von schlechten Rewards | Sichere, bekannte Rewards |
| Notwendig für Lernen | Notwendig für Performance |

**Trade-off:** Zu viel Exploration → schlechte Performance. Zu viel Exploitation → suboptimale Policy wird gelernt.

---

### 5. Monte-Carlo (MC) Methods

#### Grundidee

- Lerne **aus vollständigen Episoden**
- Warte bis Episode endet (terminaler Zustand)
- Berechne tatsächlichen Return G_t
- Update Value Function basierend auf beobachteten Returns

#### First-Visit MC Prediction

**Algorithmus:**
```
Initialisiere V(s) beliebig für alle s ∈ S
Initialisiere Returns(s) = leere Liste für alle s ∈ S

Wiederhole für jede Episode:
    Generiere Episode: S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T
    G = 0
    Für t = T-1, T-2, ..., 0:
        G = γ·G + R_{t+1}
        Wenn S_t zum ersten Mal in dieser Episode:
            Füge G zu Returns(S_t) hinzu
            V(S_t) = Durchschnitt(Returns(S_t))
```

#### Every-Visit MC

- Update bei **jedem** Besuch von s in der Episode
- First-Visit ist theoretisch sauberer, Every-Visit funktioniert in Praxis oft ähnlich

#### MC für Action Values q(s,a)

Analog zu V(s), aber für State-Action-Paare:
- Wichtig für Control (Policy-Verbesserung)
- Benötigt Exploration (z.B. ε-greedy) um alle (s,a) zu besuchen

#### MC Control mit ε-Greedy

```
Initialisiere Q(s,a) beliebig
π(s) = ε-greedy bezüglich Q

Wiederhole für jede Episode:
    Generiere Episode mit π
    Für jedes (s,a) in Episode (First-Visit):
        Berechne G_t
        Q(s,a) ← Q(s,a) + α(G_t - Q(s,a))
    Verbessere π: π(s) = ε-greedy bezüglich Q
```

#### Eigenschaften von MC

| Vorteil | Nachteil |
|---------|----------|
| Kein Bias (konvergiert zu v_π) | Hohe Varianz |
| Einfach zu implementieren | Muss bis Episode-Ende warten |
| Funktioniert auch mit nicht-Markov | Ineffizient für lange Episoden |
| Konvergiert mit Wahrscheinlichkeit 1 | Langsame Lernrate |

---

### 6. Temporal-Difference (TD) Learning

#### Grundidee

TD kombiniert Ideen aus MC und Dynamic Programming:
- **Wie MC:** Lerne direkt aus Erfahrung (ohne Modell)
- **Wie DP:** Verwende Bootstrapping (aktuelle Schätzung als Target)

#### TD(0) Prediction

**Update-Regel:**
```
V(S_t) ← V(S_t) + α · [R_{t+1} + γ·V(S_{t+1}) - V(S_t)]
```

**Komponenten:**
- **TD Target:** R_{t+1} + γ·V(S_{t+1})
- **TD Error:** δ_t = R_{t+1} + γ·V(S_{t+1}) - V(S_t)
- **Update:** V(S_t) ← V(S_t) + α · δ_t

#### TD-Algorithmus

```
Initialisiere V(s) beliebig für alle s ∈ S
Wähle Startzustand S

Wiederhole bis terminal:
    Wähle Aktion A gemäß Policy π(S)
    Führe A aus, beobachte R, S'
    TD Target: R + γ·V(S')
    TD Error: δ = R + γ·V(S') - V(S)
    V(S) ← V(S) + α·δ
    S ← S'
```

#### MC vs. TD - Direkter Vergleich

| Aspekt | Monte-Carlo | TD(0) |
|--------|-------------|-------|
| **Target** | G_t = R_{t+1} + γ·R_{t+2} + ... | R_{t+1} + γ·V(S_{t+1}) |
| **Bias** | Kein Bias | Etwas Bias (abhängig von V(S_{t+1})) |
| **Varianz** | Hoch (viele zufällige Rewards) | Niedrig (nur ein Reward) |
| **Warten** | Bis Episode-Ende | Nach jedem Schritt |
| **Bootstrapping** | Nein | Ja |
| **Konvergenz** | Konvergiert zu v_π | Konvergiert zu v_π |
| **Effizienz** | Niedrig | Hoch |

#### Warum TD oft besser ist als MC

1. **Niedrigere Varianz:** TD Target hängt nur von einem zufälligen Reward ab, MC von der gesamten Episode
2. **Schnelleres Lernen:** Updates nach jedem Schritt möglich
3. **Online-Lernen:** Funktioniert auch für continuing tasks ohne Episoden
4. **Bootstrapping:** Nutzt aktuelle Schätzungen, die bereits Information enthalten

---

### 7. SARSA (On-Policy TD Control)

#### Grundidee

SARSA lernt die Q-Function für die **aktuell verfolgte Policy**:
- **S**tate, **A**ction, **R**eward, **S**tate, **A**ction
- Die letzte Aktion A' wird tatsächlich ausgeführt

#### SARSA Update-Regel

```
Q(S_t, A_t) ← Q(S_t, A_t) + α · [R_{t+1} + γ·Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

**Wichtig:** A_{t+1} ist die **tatsächlich gewählte** Aktion im nächsten Zustand!

#### SARSA Algorithmus

```
Initialisiere Q(s,a) beliebig für alle s ∈ S, a ∈ A
Q(terminal, ·) = 0

Für jede Episode:
    Initialisiere S
    Wähle A aus S (z.B. ε-greedy bezüglich Q)
    
    Wiederhole bis S terminal:
        Führe A aus, beobachte R, S'
        Wähle A' aus S' (ε-greedy bezüglich Q)
        
        Q(S,A) ← Q(S,A) + α·[R + γ·Q(S',A') - Q(S,A)]
        
        S ← S'
        A ← A'
```

#### Eigenschaften

- **On-Policy:** Lernt über die Policy, die gerade ausgeführt wird
- **Konvergenz:** Konvergiert zu q_π für die ε-greedy Policy
- **Vorsichtig:** Berücksichtigt Explorations-Verhalten in der Q-Function

---

### 8. Q-Learning (Off-Policy TD Control)

#### Grundidee

Q-Learning lernt die **optimale Q-Function** unabhängig von der verfolgten Policy:
- Kann aus "fremden" Daten lernen
- Kann optimale Policy lernen während explorativ gehandelt wird

#### Q-Learning Update-Regel

```
Q(S_t, A_t) ← Q(S_t, A_t) + α · [R_{t+1} + γ·max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
```

**Wichtig:** max_a Q(S_{t+1}, a) - **nicht** die tatsächlich gewählte Aktion!

#### Q-Learning Algorithmus

```
Initialisiere Q(s,a) beliebig für alle s ∈ S, a ∈ A
Q(terminal, ·) = 0

Für jede Episode:
    Initialisiere S
    
    Wiederhole bis S terminal:
        Wähle A aus S (z.B. ε-greedy bezüglich Q)
        Führe A aus, beobachte R, S'
        
        Q(S,A) ← Q(S,A) + α·[R + γ·max_a Q(S',a) - Q(S,A)]
        
        S ← S'
```

#### Konvergenz

Q-Learning konvergiert mit Wahrscheinlichkeit 1 zu q* unter folgenden Bedingungen:
1. Alle (s,a)-Paare werden unendlich oft besucht
2. Learning Rate α_t(s,a) erfüllt:
   - Σ_t α_t(s,a) = ∞ (groß genug für Lernen)
   - Σ_t α_t(s,a)² < ∞ (klein genug für Konvergenz)

#### SARSA vs. Q-Learning

| Aspekt | SARSA | Q-Learning |
|--------|-------|------------|
| **Policy-Typ** | On-Policy | Off-Policy |
| **Update** | R + γ·Q(S',A') | R + γ·max_a Q(S',a) |
| **Lernt** | Wert der aktuellen Policy | Wert der optimalen Policy |
| **Exploration** | In Q-Werten berücksichtigt | Separat von Q-Learning |
| **Verhalten** | Vorsichtig (berücksichtigt ε) | Optimistisch (ignoriert ε) |
| **Beispiel** | Lernt, Cliff zu vermeiden | Lernt optimalen Pfad am Cliff |

**Cliff Walking Beispiel:**
- SARSA: Lernt sicheren Pfad (weg vom Cliff)
- Q-Learning: Lernt optimalen Pfad (entlang des Cliffs)

---

### 9. ε-Greedy Exploration

#### Grundidee

Einfache aber effektive Explorationsstrategie:

```
π(a|s) = { 1-ε + ε/|A(s)|   wenn a = argmax_a' Q(s,a')
         { ε/|A(s)|         sonst
```

- Mit Wahrscheinlichkeit **1-ε**: Greedy Aktion (beste bekannte)
- Mit Wahrscheinlichkeit **ε**: Zufällige Aktion (Exploration)

#### Typische Werte

- ε = 0.1 oder 0.05 für konstantes ε
- ε-Decay: Start mit ε=1.0, reduziere langsam auf ε_min

#### ε-Greedy in Q-Learning

```
Wenn random() < ε:
    A = zufällige Aktion
Sonst:
    A = argmax_a Q(S,a)
```

---

### 10. Deep Q-Networks (DQN)

#### Problem bei großen State Spaces

- Q-Tabelle wird zu groß (z.B. Bilder: 10^6 Pixel → 256^(10^6) Zustände)
- Lösung: **Funktionsapproximation** mit neuronalem Netz

#### DQN Architektur

```
Input (State s) → Neuronales Netz → Output (Q-Werte für alle Aktionen)
```

- **Input:** Zustand (z.B. Bild, Sensorwerte)
- **Output:** Q(s,a;w) für alle a ∈ A
- **Parameter:** Gewichte w des neuronalen Netzes

#### DQN Loss Function

```
L(w) = E[(r + γ·max_a' Q_target(s',a';w⁻) - Q(s,a;w))²]
```

- **TD Target:** r + γ·max_a' Q_target(s',a';w⁻)
- **TD Error:** TD Target - Q(s,a;w)
- **w⁻:** Parameter des Target Networks

#### Gradient Descent Update

```
Δw = α · [r + γ·max_a' Q_target(s',a';w⁻) - Q(s,a;w)] · ∇_w Q(s,a;w)
```

#### Kernkomponenten von DQN

**1. Experience Replay**

- Speichere Transitionen im Replay Buffer: (s_t, a_t, r_{t+1}, s_{t+1})
- Sample zufällige Mini-Batches aus Buffer für Training
- **Vorteile:**
  - Bricht zeitliche Korrelationen zwischen aufeinanderfolgenden Samples
  - Höhere Daten-Effizienz (jede Transition mehrfach verwendbar)
  - Stabileres Training

**2. Target Networks**

- Zwei separate Netzwerke:
  - **Q-Network:** Wird bei jedem Schritt aktualisiert
  - **Target Network:** Wird nur alle C Schritte aktualisiert
- **Vorteil:** Verhindert "Moving Target" Problem
  - Ohne Target Network: Target ändert sich bei jedem Update → Instabilität
  - Mit Target Network: Target bleibt für C Schritte stabil

#### DQN Training Algorithmus

```
Initialisiere Replay Buffer D mit Kapazität N
Initialisiere Q-Network mit zufälligen Gewichten w
Initialisiere Target Network mit w⁻ = w

Für Episode = 1, ..., M:
    Initialisiere State s_1
    
    Für t = 1, ..., T:
        Wähle Aktion a_t = ε-greedy bezüglich Q(·;w)
        Führe a_t aus, beobachte r_t, s_{t+1}
        Speichere (s_t, a_t, r_t, s_{t+1}) in D
        
        Sample zufälliges Mini-Batch {(s_j, a_j, r_j, s_{j+1})} aus D
        
        Für jede Transition im Batch:
            Wenn s_{j+1} terminal:
                y_j = r_j
            Sonst:
                y_j = r_j + γ·max_a' Q_target(s_{j+1}, a'; w⁻)
        
        Führe Gradient Descent auf (y_j - Q(s_j, a_j; w))² durch
        
        Alle C Schritte: w⁻ ← w
```

#### DQN Erfolge

- **Atari Games:** DQN erreichte menschliches oder übermenschliches Niveau in vielen Atari-Spielen
- **End-to-End:** Lernt direkt aus Pixel-Input
- **General Purpose:** Selbe Architektur für alle Spiele

---

### 11. Double DQN

#### Problem: Overestimation Bias

DQN neigt zur **systematischen Überschätzung** von Q-Werten:

**Ursache:**
```
Q-Learning verwendet max_a Q(s',a) für:
1. Aktion-Selektion: Welche Aktion ist die beste?
2. Aktion-Bewertung: Wie gut ist diese Aktion?
```

- Derselbe Q-Wert wird für beides verwendet
- Maximierung über noisy Schätzungen → positive Verzerrung
- "Maximization Bias": max von noisy Werten ist im Erwartungswert größer als max der wahren Werte

#### Double DQN Lösung

**Entkopplung von Selektion und Bewertung:**

1. **Selektion:** Welche Aktion ist die beste?
   - Verwende Q-Network: a* = argmax_a Q(s', a; w)

2. **Bewertung:** Wie gut ist diese Aktion?
   - Verwende Target Network: Q_target(s', a*; w⁻)

#### Double DQN Update

```
# Normales DQN:
Q_target_DQN = r + γ · max_a' Q_target(s', a'; w⁻)

# Double DQN:
a* = argmax_a' Q(s', a'; w)              # Selektion mit Q-Network
Q_target_Double = r + γ · Q_target(s', a*; w⁻)  # Bewertung mit Target Network
```

#### Eigenschaften

- **Reduziert Overestimation:** Deutlich geringere Q-Wert-Überschätzung
- **Bessere Performance:** Stabileres Training, bessere Policies
- **Einfache Implementierung:** Kleine Änderung am DQN-Code

---

### 12. Function Approximation

#### Grundidee

Statt Q-Tabelle: approximiere Q-Funktion mit parametrisierter Funktion:

```
Q(s,a) ≈ q(s,a;w)
```

- **w:** Parameter (z.B. Gewichte eines neuronalen Netzes)
- **Vorteil:** Generalisierung von gesehenen zu ungesehenen Zuständen

#### Gradient Descent für Value Functions

Für State-Value Function V(s;w):

```
Δw = α · [G_t - V(S_t;w)] · ∇_w V(S_t;w)
```

Für Action-Value Function Q(s,a;w):

```
Δw = α · [TD Target - Q(S_t,A_t;w)] · ∇_w Q(S_t,A_t;w)
```

#### Semi-Gradient Methods

- **Problem:** TD Target hängt von w ab (über Q(s',a';w))
- **Lösung:** Ignoriere diese Abhängigkeit im Gradienten
- **Name:** "Semi-Gradient" weil nicht der volle Gradient
- **Konvergenz:** Nicht garantiert für off-policy, aber funktioniert in Praxis

---

### 13. On-Policy vs. Off-Policy

#### On-Policy Methods

- **Definition:** Lernt über die Policy π, die gerade ausgeführt wird
- **Beispiele:** SARSA, MC Control
- **Eigenschaften:**
  - Kann nicht aus fremden Daten lernen
  - Berücksichtigt Explorations-Verhalten
  - Oft vorsichtigeres Verhalten

#### Off-Policy Methods

- **Definition:** Lernt über eine Policy π (target policy) während eine andere Policy μ (behavior policy) ausgeführt wird
- **Beispiele:** Q-Learning, DQN
- **Eigenschaften:**
  - Kann aus beliebigen Daten lernen (auch von anderen Agenten)
  - Kann optimale Policy lernen während explorativ gehandelt wird
  - Effizientere Datennutzung

#### Vergleich

| On-Policy | Off-Policy |
|-----------|------------|
| SARSA | Q-Learning |
| Lernt π während π ausgeführt wird | Lernt π* während μ ausgeführt wird |
| Kann nicht aus alten Daten lernen | Kann aus Experience Replay lernen |
| Berücksichtigt Exploration | Separat von Exploration |
| Vorsichtig | Optimistisch |

---

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ Q-Learning Update (SEHR WICHTIG)

**Warum:** Grundlegender Off-Policy Algorithmus, Basis für DQN

**Formel:**
```
Q(s,a) ← Q(s,a) + α · [r + γ·max_a' Q(s',a') - Q(s,a)]
```

**Wichtig zu verstehen:**
- max_a' Q(s',a') - nicht die tatsächlich gewählte Aktion!
- TD Target: r + γ·max_a' Q(s',a')
- TD Error: TD Target - Q(s,a)

### ✅ DQN (Deep Q-Networks) (SEHR WICHTIG)

**Warum:** Erfolgreiche Deep RL Methode, verbindet RL mit Deep Learning

**Kernkomponenten:**
1. **Experience Replay:** Speichert und resamplet Transitionen
2. **Target Networks:** Stabilisiert Training durch separates Target Network

**Loss Function:**
```
L(w) = E[(r + γ·max_a' Q_target(s',a';w⁻) - Q(s,a;w))²]
```

### ✅ Double DQN (SEHR WICHTIG)

**Warum:** Löst Overestimation Problem von DQN

**Update:**
```
a* = argmax_a' Q(s',a';w)                    # Selektion
Q_double = r + γ·Q_target(s',a*;w⁻)          # Bewertung
```

**Unterschied zu DQN:**
- DQN: max_a' Q_target(s',a';w⁻)
- Double DQN: Q_target(s', argmax_a' Q(s',a';w); w⁻)

### ✅ Bellman-Gleichungen (WICHTIG)

**Warum:** Mathematisches Fundament von RL

**Bellman-Optimality für q*:**
```
q*(s,a) = E[r + γ·max_a' q*(s',a')]
```

**Verständnis:** Rekursive Struktur - Wert von (s,a) = sofortiger Reward + diskontierter Wert der besten nächsten Aktion

### ✅ MC vs. TD (WICHTIG)

**Warum:** Zwei grundlegende Ansätze zum Lernen von Value Functions

| MC | TD |
|----|-----|
| Kein Bias, hohe Varianz | Etwas Bias, niedrige Varianz |
| Wartet bis Episode-Ende | Update nach jedem Schritt |
| Kein Bootstrapping | Bootstrapping |

### ✅ SARSA vs. Q-Learning (WICHTIG)

**Warum:** On-Policy vs. Off-Policy

| SARSA | Q-Learning |
|-------|------------|
| R + γ·Q(S',A') | R + γ·max_a Q(S',a) |
| On-Policy | Off-Policy |
| Lernt aktuelle Policy | Lernt optimale Policy |

---

## Formeln/Algorithmen (auswendig können!)

### Return
```
G_t = Σ_{k=0}^{∞} γ^k · r_{t+k+1}
```

### State-Value Function
```
v_π(s) = E_π[G_t | S_t = s]
```

### Action-Value Function
```
q_π(s,a) = E_π[G_t | S_t = s, A_t = a]
```

### Bellman-Optimality für q*
```
q*(s,a) = E[r + γ·max_a' q*(s',a')]
```

### SARSA Update
```
Q(s,a) ← Q(s,a) + α · [r + γ·Q(s',a') - Q(s,a)]
```

### Q-Learning Update
```
Q(s,a) ← Q(s,a) + α · [r + γ·max_a' Q(s',a') - Q(s,a)]
```

### TD Error
```
δ_t = R_{t+1} + γ·V(S_{t+1}) - V(S_t)
```

### DQN Loss
```
L(w) = E[(r + γ·max_a' Q_target(s',a';w⁻) - Q(s,a;w))²]
```

### Double DQN
```
a* = argmax_a Q(s',a;w)
Q_double = r + γ·Q_target(s',a*;w⁻)
```

### ε-Greedy Policy
```
π(a|s) = { 1-ε + ε/|A|   wenn a = argmax Q(s,a')
         { ε/|A|         sonst
```

---

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte

1. **RL Ziel:** Maximiere erwarteten kumulativen Return G_t
2. **Value Functions:** v_π(s) und q_π(s,a) quantifizieren "Güte" von Zuständen/Aktionen
3. **Bellman-Gleichungen:** Rekursive Struktur ermöglicht iterative Lösung
4. **MC vs. TD:** MC = kein Bias, hohe Varianz; TD = etwas Bias, niedrige Varianz
5. **SARSA:** On-Policy, lernt Wert der aktuellen Policy
6. **Q-Learning:** Off-Policy, lernt Wert der optimalen Policy
7. **DQN:** Skaliert auf große State Spaces durch Deep Learning
8. **Experience Replay:** Bricht Korrelationen, höhere Daten-Effizienz
9. **Target Networks:** Verhindert "Moving Target" Problem
10. **Double DQN:** Behebt systematische Q-Wert-Überschätzung

### ⚠️ Häufige Fehler

1. **Q-Learning vs. SARSA verwechseln:**
   - Q-Learning: max_a' Q(s',a') - optimale nächste Aktion
   - SARSA: Q(s',a') - tatsächlich gewählte nächste Aktion

2. **Target Network vergessen:** Ohne Target Network divergiert DQN oft

3. **Learning Rate zu hoch:** Führt zu Instabilität und Divergenz

4. **Exploration vernachlässigen:** Ohne Exploration wird suboptimale Policy gelernt

5. **Overestimation ignorieren:** DQN überschätzt systematisch Q-Werte

### 📝 Prüfungsrelevante Fragen

1. **Was ist der Unterschied zwischen MC und TD?**
   - MC: Kein Bias, hohe Varianz, wartet bis Episode-Ende, kein Bootstrapping
   - TD: Etwas Bias, niedrige Varianz, Update nach jedem Schritt, Bootstrapping

2. **Wie funktioniert Q-Learning?**
   - Off-Policy TD Control Algorithmus
   - Update: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
   - Lernt optimale Policy unabhängig von verfolgter Policy

3. **Was ist der Vorteil von Double DQN gegenüber DQN?**
   - Reduziert Overestimation Bias
   - Entkoppelt Aktion-Selektion von Aktion-Bewertung

4. **Was ist Experience Replay und warum ist es wichtig?**
   - Speichert Transitionen (s,a,r,s') im Buffer
   - Zufälliges Sampling für Training
   - Bricht zeitliche Korrelationen, höhere Daten-Effizienz

5. **Was ist der Unterschied zwischen On-Policy und Off-Policy?**
   - On-Policy: Lernt über Policy die ausgeführt wird (SARSA)
   - Off-Policy: Lernt über andere Policy als die ausgeführte (Q-Learning)

6. **Wie funktionieren Target Networks?**
   - Separates Network für TD Target
   - Wird nur periodisch aktualisiert
   - Verhindert "Moving Target" Problem, stabileres Training

7. **Was ist der Unterschied zwischen SARSA und Q-Learning?**
   - SARSA: R + γ·Q(S',A') - verwendet tatsächlich gewählte Aktion
   - Q-Learning: R + γ·max_a Q(S',a) - verwendet beste Aktion

8. **Was ist das Overestimation Problem?**
   - DQN überschätzt systematisch Q-Werte
   - Ursache: max Operator auf noisy Schätzungen
   - Lösung: Double DQN entkoppelt Selektion und Bewertung

9. **Was ist die Bellman-Optimality-Gleichung für q*?**
   - q*(s,a) = E[r + γ·max_a' q*(s',a')]
   - Fundament für Q-Learning

10. **Was ist ε-Greedy Exploration?**
    - Mit Wahrscheinlichkeit 1-ε: Greedy Aktion
    - Mit Wahrscheinlichkeit ε: Zufällige Aktion
    - Balanciert Exploration und Exploitation

---

## Zusammenfassung der wichtigsten Konzepte

| Konzept | Beschreibung | Formel/Algorithmus |
|---------|-------------|-------------------|
| **Return** | Kumulativer, diskontierter Reward | G_t = Σ γ^k · r_{t+k+1} |
| **Q-Function** | Erwarteter Return für (s,a) | q_π(s,a) = E_π[G_t \| S_t=s, A_t=a] |
| **Bellman-Optimality** | Rekursive Struktur optimaler Werte | q*(s,a) = E[r + γ·max_a' q*(s',a')] |
| **Q-Learning** | Off-Policy TD Control | Q ← Q + α[r + γ·max Q' - Q] |
| **SARSA** | On-Policy TD Control | Q ← Q + α[r + γ·Q' - Q] |
| **DQN** | Deep Learning + Q-Learning | Experience Replay + Target Networks |
| **Double DQN** | Reduziert Overestimation | Trennt Selektion und Bewertung |

---

## Lernempfehlungen für die Klausur

### Priorität 1 (SEHR WICHTIG):
- ✅ Q-Learning Update-Regel verstehen und anwenden können
- ✅ Unterschied Q-Learning vs. SARSA erklären können
- ✅ DQN Komponenten (Experience Replay, Target Networks) verstehen
- ✅ Double DQN Update und Vorteil gegenüber DQN

### Priorität 2 (WICHTIG):
- ✅ Bellman-Gleichungen für v_π und q_π
- ✅ MC vs. TD Vor-/Nachteile
- ✅ On-Policy vs. Off-Policy Unterschied
- ✅ ε-Greedy Exploration

### Priorität 3 (GRUNDWISSEN):
- ✅ RL Grundkonzepte (Agent, Environment, State, Action, Reward)
- ✅ Return und Discount-Faktor
- ✅ Value Functions Definitionen

---

**Erstellt:** 17.03.2026
**Basierend auf:** AdvancedML-07-RL-Teil-1.pdf (~90 Seiten)
**Klausurrelevanz:** SEHR HOCH - RL-Teil ist einer der wichtigsten Bereiche
