# ZUSAMMENFASSUNG 07: Reinforcement Learning Teil 1

## Übersicht
- Seitenzahl: ~90 Seiten
- Hauptthemen: RL-Grundlagen, Value-Based Methods, Q-Learning, DQN, Double DQN

## Detaillierte Inhalte

### 1. RL-Grundlagen

#### Kernkonzepte
- **Agent:** Entscheidungs-Einheit
- **Environment:** Umgebung, in der Agent agiert
- **State (s_t):** Zustand zum Zeitpunkt t
- **Action (a_t):** Aktion zum Zeitpunkt t
- **Reward (r_t):** Skalares Feedback-Signal

#### Policy
```
a_t = π(s_t)          (deterministisch)
a_t ~ π(·|s_t)        (stochastisch)
```
- Definiert Verhalten des Agenten

#### Return
```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...
```
- **γ (Gamma):** Discount-Faktor ∈ [0,1]
- Kumulativer, diskontierter Reward ab Zeitpunkt t

#### Reward-Hypothese
> Jedes Ziel kann als Maximierung eines kumulativen Rewards formalisiert werden.

### 2. Value Functions

#### State-Value Function
```
v_π(s) = E_π[G_t | S_t = s]
       = E_π[r_{t+1} + γ·v_π(S_{t+1}) | S_t = s]
```
- Erwarteter Return ausgehend von Zustand s unter Policy π

#### Action-Value Function (Q-Function)
```
q_π(s,a) = E_π[G_t | S_t = s, A_t = a]
```
- Erwarteter Return bei Ausführung von Aktion a in Zustand s

#### Optimal Value Functions
```
v*(s) = max_a E[r_{t+1} + γ·v*(S_{t+1}) | S_t = s, A_t = a]
q*(s,a) = E[r_{t+1} + γ·max_a' q*(S_{t+1}, a') | S_t = s, A_t = a]
```

### 3. Bellman-Gleichungen

#### Bellman-Expectation
```
v_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γ·v_π(s')]
```

#### Bellman-Optimality
```
v*(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γ·v*(s')]
q*(s,a) = Σ_{s',r} p(s',r|s,a)[r + γ·max_{a'} q*(s',a')]
```

### 4. Exploration vs Exploitation

- **Exploration:** Neue Dinge ausprobieren
- **Exploitation:** Bekannte gute Aktionen nutzen
- Beides wichtig, Trade-off

### 5. Monte-Carlo (MC) Methods

#### Grundidee
- Lerne aus vollständigen Episoden
- Value = durchschnittlicher Return

#### First-Visit MC
```
V(s) = V(s) + α(G_t - V(s))
```
- Update nur beim ersten Besuch von s in Episode

#### Every-Visit MC
- Update bei jedem Besuch von s

#### Eigenschaften
- **Vorteil:** Kein Bias, konvergiert zu v_π
- **Nachteil:** Hohe Varianz, muss bis Episode-Ende warten

### 6. Temporal-Difference (TD) Learning

#### TD(0) Update
```
V(S_t) ← V(S_t) + α[R_{t+1} + γ·V(S_{t+1}) - V(S_t)]
```
- **TD Target:** R_{t+1} + γ·V(S_{t+1})
- **TD Error:** δ_t = R_{t+1} + γ·V(S_{t+1}) - V(S_t)

#### MC vs TD
| Aspekt | MC | TD |
|--------|-----|-----|
| Bias | Kein Bias | Etwas Bias |
| Varianz | Hoch | Niedrig |
| Warten | Bis Episode-Ende | Nach jedem Schritt |
| Bootstrapping | Nein | Ja |

### 7. SARSA (On-Policy)

#### Update-Regel
```
Q(S,A) ← Q(S,A) + α[R + γ·Q(S',A') - Q(S,A)]
```
- **On-Policy:** Lernt über Policy π während Aktionen gemäß π gewählt
- A' wird gemäß aktueller Policy gewählt

#### Algorithmus
1. Initialisiere Q
2. Für jede Episode:
3. Initialisiere S, wähle A aus S (ε-greedy)
4. Wiederhole bis Ende:
   - Führe A aus, beobachte R, S'
   - Wähle A' aus S' (ε-greedy)
   - Q(S,A) ← Q(S,A) + α[R + γ·Q(S',A') - Q(S,A)]
   - S ← S', A ← A'

### 8. Q-Learning (Off-Policy)

#### Update-Regel
```
Q(S,A) ← Q(S,A) + α[R + γ·max_a' Q(S',a') - Q(S,A)]
```

#### Eigenschaften
- **Off-Policy:** Kann optimale Policy lernen während explorative Policy verfolgt
- Max über alle Aktionen in S' (nicht die tatsächlich gewählte)

#### Konvergenz
- Konvergiert zu q* mit Wahrscheinlichkeit 1
- Unter Annahmen: alle (s,a) besucht unendlich oft, α abnehmend

### 9. ε-Greedy Exploration

```
π(a|s) = { 1-ε + ε/|A|  if a = argmax Q(s,a')
         { ε/|A|        sonst
```
- Mit Wahrscheinlichkeit 1-ε: Greedy Aktion
- Mit Wahrscheinlichkeit ε: Zufällige Aktion

### 10. Deep Q-Networks (DQN)

#### Problem bei großen State Spaces
- Q-Tabelle zu groß
- Lösung: Neuronales Netz als Funktionsapproximator

#### DQN Architektur
- Input: Zustand (z.B. Bild)
- Output: Q-Werte für alle Aktionen
- **Q-Netzwerk:** Q(s,a;w)

#### Kernkomponenten

**1. Experience Replay**
- Speichere Transitionen (s, a, r, s') im Buffer
- Sample zufällig aus Buffer für Training
- Bricht Korrelationen zwischen aufeinanderfolgenden Samples

**2. Target Networks**
- Zwei Netzwerke: Q (aktuell) und Q_target
- Q_target wird nur periodisch aktualisiert
- Verhindert "Moving Target" Problem

#### DQN Loss
```
L(w) = E[(r + γ·max_a' Q_target(s',a';w⁻) - Q(s,a;w))²]
```

#### Training
```
Δw = α·[r + γ·max_a' Q_target(s',a') - Q(s,a)]·∇_w Q(s,a)
```

### 11. Double DQN

#### Problem: Overestimation
- DQN überschätzt Q-Werte
- max Operator verwendet dieselben Werte für Selektion und Bewertung

#### Double DQN Lösung
- Entkopplung von Aktion-Selektion und Aktion-Bewertung
- Aktion selektieren mit Q-Netzwerk
- Aktion bewerten mit Target-Netzwerk

#### Update
```
a* = argmax_a' Q(s',a';w)           # Selektion mit Q
Q_double(s,a) = r + γ·Q_target(s',a*;w⁻)  # Bewertung mit Target
```

### 12. Function Approximation

#### Grundidee
- V(s) ≈ v(s;w) oder Q(s,a) ≈ q(s,a;w)
- Generalisierung von gesehenen zu ungesehenen Zuständen

#### Gradient Descent für Value Functions
```
Δw = α·[G_t - v(S_t;w)]·∇_w v(S_t;w)
```

### 13. Replay Buffer

- Speichert Transitionen: (s_t, a_t, r_{t+1}, s_{t+1})
- Zufälliges Sampling für Training
- Höhere Daten-Effizienz
- Größere Stabilität

### 14. On-Policy vs Off-Policy

| On-Policy | Off-Policy |
|-----------|------------|
| Lernt über π während π ausgeführt wird | Lernt über π während μ ausgeführt |
| SARSA | Q-Learning, DQN |
| Kann nicht aus fremden Daten lernen | Kann aus beliebigen Daten lernen |

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ Q-Learning Update
- Warum: Grundlegender Off-Policy Algorithmus
- Was: max_a' Q(s',a'), TD-Target

### ✅ DQN (Deep Q-Networks)
- Warum: Erfolgreiche Deep RL Methode
- Was: Experience Replay, Target Networks

### ✅ Double DQN
- Warum: Löst Overestimation Problem
- Was: Entkopplung von Selektion und Bewertung

### ✅ Bellman-Gleichungen
- Warum: Mathematisches Fundament
- Was: Rekursive Struktur, Optimalität

### ✅ MC vs TD
- Warum: Zwei grundlegende Ansätze
- Was: Bias-Variance Tradeoff, Bootstrapping

## Formeln/Algorithmen (wichtig)

### Return
```
G_t = Σ_{k=0}^∞ γ^k · r_{t+k+1}
```

### Q-Learning Update
```
Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
```

### SARSA Update
```
Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
```

### Bellman-Optimalitäts-Gleichung
```
q*(s,a) = E[r + γ·max_a' q*(s',a')]
```

### DQN Loss
```
L = (r + γ·max_a' Q_target(s',a') - Q(s,a))²
```

### Double DQN
```
a* = argmax_a Q(s',a;w)
Q_double = r + γ·Q_target(s',a*;w⁻)
```

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte
- **RL Ziel:** Maximiere erwarteten Return
- **Q-Learning:** Off-Policy, lernt optimale Aktionen während explorierend
- **DQN:** Skaliert auf große State Spaces durch Deep Learning
- **Target Networks:** Wichtig für stabiles Training
- **Double DQN:** Behebt systematische Überschätzung

### ⚠️ Häufige Fehler
- Q-Learning: max über nächsten Zustand vs SARSA: tatsächliche Aktion
- Target Network nicht aktualisieren
- Learning Rate zu hoch → Divergenz

### 📝 Prüfungsrelevante Fragen
1. Was ist der Unterschied zwischen MC und TD?
2. Wie funktioniert Q-Learning?
3. Was ist der Vorteil von Double DQN gegenüber DQN?
4. Was ist Experience Replay und warum ist es wichtig?
5. Was ist der Unterschied zwischen On-Policy und Off-Policy?
6. Wie funktionieren Target Networks?
7. Was ist der Unterschied zwischen SARSA und Q-Learning?
8. Was ist das Overestimation Problem?
