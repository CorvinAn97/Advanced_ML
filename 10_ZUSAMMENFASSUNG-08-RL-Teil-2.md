# ZUSAMMENFASSUNG 08: Reinforcement Learning Teil 2

## Übersicht
- Seitenzahl: ~75 Seiten
- Hauptthemen: Exploration, Offline RL, Actor-Critic, Policy Gradients (nur grundlegend)

## Detaillierte Inhalte

### 1. Exploration in RL

#### Warum ist Exploration schwierig?
- Agent kennt Regeln nicht
- Kann Regeln nur durch Ausprobieren herausfinden
- Zeitlich ausgedehnte Aufgaben → schwieriger
- Beispiel: Montezuma's Revenge - Schlüssel sammeln → Reward, aber erst nach vielen Schritten

#### Exploration vs Exploitation
- **Exploitation:** Tun, was aktuell für das Beste gehalten wird
- **Exploration:** Neue Dinge ausprobieren
- Beides wichtig, Trade-off

### 2. Bandits (Einführung)

#### One-Armed Bandit
- Eine Aktion (Arm ziehen)
- Reward unbekannt

#### Multi-Armed Bandit
- K Arme
- Jeder Arm a_i hat eigene Reward-Verteilung p(r|a_i)
- Einfachstes Explorationsproblem

#### Regret
```
Reg(T) = T·E[r(a*)] - Σ_{t=1}^T E[r(a_t)]
```
- Differenz zur optimalen Policy

### 3. Exploration-Strategien für Bandits

#### 1. Upper Confidence Bound (UCB)
**Idee:** Versuche jeden Arm bis es sicher ist, dass er nicht gut ist

```
a = argmax_a [μ̂_a + C·σ_a]
```
- μ̂_a: Geschätzter durchschnittlicher Reward
- σ_a: Unsicherheit (z.B. √(2·ln(T)/N(a)))
- C: Explorations-Parameter

**Spezifische Form (Auer et al.):**
```
a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]
```
- Regret: O(log T) - beweisbar optimal

**Intuition:**
- Bei Unsicherheit → Optimismus
- Mehr Exploration bei weniger gesehenen Aktionen

#### 2. Thompson Sampling
**Idee:** Sample aus Posterior-Verteilung

```
θ_1, ..., θ_K ~ p̂(θ_1, ..., θ_K)
a = argmax_a E_θ_a[r|a]
```

**Vorgehen:**
1. Sample Parameter θ aus Posterior
2. Nimm an θ sei korrekt
3. Wähle optimale Aktion für θ
4. Update Posterior aus Beobachtung

#### 3. Information Gain
**Idee:** Wähle Aktion mit maximalem Informationsgewinn

```
IG(z,y|a) = E_y[H(p̂(z)) - H(p̂(z|y))|a]
```

- z: Latente Variable (z.B. optimale Aktion, Parameter)
- y: Beobachtung
- H: Entropie

**Beispiel:**
- a* = argmin_a [Δ_a² / g_a]
- Δ_a: Erwartete Suboptimalität
- g_a: Information Gain

### 4. Exploration in Deep RL

#### Optimistische Exploration
- Neuer Zustand = guter Zustand
- Explorations-Bonus: r⁺(s,a) = r(s,a) + B(N(s))
- B(N(s)) abnehmend mit Besuchszahl

#### Problem: Wie zählen wir Zustände?
- Riesige/kontinuierliche Zustandsräume
- Lösung: Pseudo-Counts

### 5. Pseudo-Counts

#### Idee
- Fittes Modell p_θ(s) für Zustandsdichte
- Nutze als "Pseudo-Count" Ñ(s)

#### Berechnung
```
Nach Beobachtung s:
p_θ(s) = n/N
p_θ'(s) = (n+1)/(N+1)
```

- Löse nach n und N auf
- Ñ(s) = n ≈ p_θ(s) / (p_θ'(s) - p_θ(s))

#### Bonus-Funktionen
- **UCB:** B(N) = √(2·ln(n)/N)
- **MBIE-EB:** B(N) = 1/N
- **BEB:** B(N) = 1/√N

### 6. Weitere Exploration-Methoden

#### Counting mit Hashes
- Komprimiere s in k-bit Code φ(s)
- Zähle N(φ(s))

#### Exemplar-Modelle
- Zustand ist neu, wenn leicht von bisherigen unterscheidbar
- Classifier: s vs D (alle bisherigen Zustände)
- Dichte: p_θ(s) = (1 - D(s)) / D(s)

#### Fehler-basierte Exploration
```
L(s,a) = ||f_θ(s,a) - f*(s,a)||²
```
- Hoher Fehler = neuer/unbekannter Zustand
- f*: Zufällige Zielfunktion (oder Next-State Prediction)

### 7. Thompson Sampling in Deep RL

#### Bootstrapped DQN
- Multiple Q-Netzwerke mit Bootstrapping
- Sample ein Netzwerk pro Episode
- Exploration durch verschiedene Q-Funktionen

**Vorteil:**
- Kohärente Policy für ganze Episode (nicht zufällige Aktionen)

### 8. Offline Reinforcement Learning

#### Problemstellung
- Datensatz D = {(s,a,s',r)} vorhanden
- Keine weitere Interaktion mit Environment
- Lerne beste Policy aus festem Datensatz

#### Motivation
- Sicherheit: Keine riskanten Interaktionen
- Kosten: Echte Roboter teuer
- Daten-Wiederverwendung

#### Formale Definition
```
D = {(s_i, a_i, s'_i, r_i)}
s ~ d^π'(s)  (unbekannte Verteilung)
a ~ π^D(a|s)  (Daten-sammelnde Policy)
```

**Ziel:**
```
max_π E[Σ_t γ^t r(s_t, a_t)]
```
(aber nur mit Daten aus D)

#### Schwierigkeit: Distribution Shift
- Online RL: Policy probiert Aktion aus und sieht was passiert
- Offline RL: Ungesehene (out-of-distribution) Aktionen können nicht ausprobiert werden

**Problem:**
- Q-Learning maximiert über Q-Werte
- Aber: Q(s',a') für ungesehene a' unbekannt
- Q-Funktion extrapoliert → Überschätzung

### 9. Offline RL Lösungsansätze

#### 1. Conservative Q-Learning (CQL)
- Minimiere Q-Werte für ungesehene Aktionen
- Zusätzlicher Regularisierungsterm:
```
L_CQL = E_{s~D, a~π}[Q(s,a)] - E_{s,a~D}[Q(s,a)]
```

#### 2. Behavior Regularization
- Policy soll nah an Daten-Policys bleiben
- Einschränkung: π(a|s) ≈ π^D(a|s)

#### 3. Importance Sampling
- Gewichte für off-policy Evaluation
- Kann hohe Varianz haben

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ UCB (Upper Confidence Bound)
- Warum: Optimale Exploration für Bandits
- Was: μ̂ + √(2·ln(T)/N(a)), Optimismus bei Unsicherheit

### ✅ Thompson Sampling
- Warum: Elegante Bayesianische Methode
- Was: Sampling aus Posterior, probability matching

### ✅ Pseudo-Counts
- Warum: Skaliert auf große Zustandsräume
- Was: Density Model, Näherung von Besuchszahlen

### ✅ Offline RL Problem
- Warum: Wichtige praktische Anwendung
- Was: Distribution Shift, kein Online-Feedback

## Formeln/Algorithmen (wichtig)

### UCB
```
a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]
```

### Thompson Sampling
```
θ ~ p̂(θ)
a = argmax_a E_θ[r|a]
```

### Pseudo-Count
```
Ñ(s) = p_θ(s) / (p_θ'(s) - p_θ(s))
```

### Explorations-Bonus
```
r⁺(s,a) = r(s,a) + B(N(s))
B(N) = 1/√N  (oder andere Varianten)
```

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte
- **Bandits:** Einfachste Explorationsprobleme, gut verstanden
- **UCB:** Optimistisch bei Unsicherheit, optimal für Bandits
- **Thompson Sampling:** Bayesianisch, gut empirisch
- **Pseudo-Counts:** Zählen in hochdimensionalen Räumen
- **Offline RL:** Lernen ohne Interaktion, Distribution Shift Problem

### ⚠️ Häufige Fehler
- UCB-Formel verwechseln
- Pseudo-Count Berechnung
- Nicht verstehen, warum Offline RL schwierig ist

### 📝 Prüfungsrelevante Fragen
1. Was ist der Unterschied zwischen Exploration und Exploitation?
2. Wie funktioniert UCB?
3. Was ist Thompson Sampling?
4. Was sind Pseudo-Counts und warum braucht man sie?
5. Was ist das Hauptproblem bei Offline RL?
6. Wie funktioniert optimistische Exploration?
7. Was ist der Unterschied zwischen Bandits und vollen MDPs?
