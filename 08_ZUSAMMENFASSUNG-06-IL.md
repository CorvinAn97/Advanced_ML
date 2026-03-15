# ZUSAMMENFASSUNG 06: Imitation Learning

## Übersicht
- Seitenzahl: ~40 Seiten
- Hauptthemen: Behavioral Cloning, Distributional Shift, DAgger, Goal-Conditioned IL

## Detaillierte Inhalte

### 1. Behavioral Cloning (Grundkonzept)

#### Idee
- Lerne Policy direkt von Experten-Demonstrationen
- Supervised Learning: Input = Zustand, Target = Experten-Aktion

#### Formel
```
π_θ(a|o) ≈ π*(a|o)
```

#### Training
```
max_θ E_{(o,a)~D_demo}[log π_θ(a|o)]
```

#### Problem: Distributional Shift
- Gelernte Policy macht Fehler
- Fehler bringen Agenten in Zustände, die nicht in Trainingsdaten enthalten
- Mit jedem Schritt werden Fehler größer
- **i.i.d. Annahme verletzt!**

### 2. Distributional Shift - Formal

#### Trainings- vs Test-Verteilung
```
p_train(o_t) ≠ p_test(o_t)
```

#### Worst-Case Analyse
- Angenommen: π_θ(a ≠ π*(s)|s) ≤ ε
- Für Zustände außerhalb D_train:
```
E[c(s_t, a_t)] = O(εT)
```
- **T Terme, jeder O(ε)** → Fehler akkumulieren sich!

#### Warum funktioniert es manchmal?
- NVIDIA DRIVES (2016): Funktionierte durch Data Augmentation
- Seitwärts gerichtete Kameras simulieren Fehler und Korrekturen
- Fehler + Korrekturen in Trainingsdaten

### 3. Lösungen für Distributional Shift

#### 1. Daten-Augmentation
- "Fake" Daten mit simulierten Fehlern
- Seitliche Kamerabilder = simulierte Abweichung

#### 2. Mächtige Modelle
- Deep Learning Modelle machen weniger Fehler
- Bessere Generalisierung

#### 3. Multi-Task-Learning
- Goal-Conditioned Behavioral Cloning
- Policy π(a|s, g) für beliebige Ziele g
- Mehr verschiedene Zustände → bessere Generalisierung

#### 4. Algorithmus anpassen: DAgger

### 4. DAgger (Dataset Aggregation)

#### Problem
- p_train(o_t) ≠ p_test(o_t)
- Idee: Passe p_train an, um p_test abzudecken

#### Algorithmus
1. Trainiere π_θ auf Experten-Daten D
2. Führe π_θ aus, erhalte Datensatz D_π
3. Experte labelt D_π mit korrekten Aktionen
4. Aggregiere: D ← D ∪ D_π
5. Wiederhole

#### Vorteile
- Keine Änderung der Policy-Architektur nötig
- Theoretische Garantie: O(T) statt O(εT²)

#### Nachteile
- Experte muss online verfügbar sein
- Kann unnatürlich sein, nachträglich zu labeln

### 5. Warum macht Policy Fehler?

#### 1. Experten-Verhalten nicht Markov
- Menschen berücksichtigen History
- Policy π(a|o) sieht nur aktuelle Beobachtung
- **Lösung:** RNN/LSTM nutzen: π(a|o_1, ..., o_t)

#### 2. Multimodales Verhalten
- Mehrere korrekte Aktionen möglich
- Einfache Wahrscheinlichkeitsverteilung: Mittelwert von zwei guten Aktionen = schlechte Aktion
- **Lösung:** Ausdrucksstärkere Verteilungen (Mixture Models, GMM)

### 6. Goal-Conditioned Behavioral Cloning

#### Idee
- Lernen, beliebige Ziele zu erreichen
- π(a|s, g) mit Ziel-Parameter g

#### Training
- Mehrere Demos mit verschiedenen Zielen
- Für jede Demo: (s_0, a_0, ..., s_T) mit Ziel g = s_T
- Maximiere: log π(a_t | s_t, g=s_T)

#### Vorteile
- Mehr verschiedene Zustände in Daten
- Bessere Generalisierung
- Beispiele: Learning Latent Plans from Play

### 7. Automatisierte Datensammlung

#### Learning to Reach Goals via Iterated Supervised Learning
1. Starte mit Zufallspolicy
2. Sammle Daten mit zufälligen Zielen
3. Behavioral Cloning auf erreichten Zielen
4. Nutze um Policy zu verbessern
5. Wiederhole

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ Distributional Shift Problem
- Warum: Kernproblem von Behavioral Cloning
- Was: p_train ≠ p_test, Fehler-Akkumulation

### ✅ DAgger Algorithmus
- Warum: Wichtige Lösung für Distributional Shift
- Was: Dataset Aggregation, iterative Verbesserung

### ✅ Goal-Conditioned IL
- Warum: Moderne Alternative
- Was: Multi-Task, bessere Generalisierung

## Formeln/Algorithmen (wichtig)

### Behavioral Cloning Loss
```
L = -E[log π_θ(a|o)]
```

### DAgger Algorithmus
```
for iteration i:
    D_i = rollout(π_i)  # Ausführen der aktuellen Policy
    D_i^labeled = expert_label(D_i)
    D = D ∪ D_i^labeled
    π_{i+1} = train(D)
```

### Goal-Conditioned BC
```
L = -Σ_t log π(a_t | s_t, g=s_T)
```

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte
- **Behavioral Cloning:** Einfach aber Distributional Shift Problem
- **DAgger:** Löst Problem durch iterative Datensammlung
- **Goal-Conditioned:** Multi-Task Ansatz für bessere Generalisierung

### ⚠️ Häufige Fehler
- Nicht erkennen, dass i.i.d. Annahme verletzt ist
- DAgger vs Behavioral Cloning verwechseln

### 📝 Prüfungsrelevante Fragen
1. Was ist Distributional Shift und warum ist es ein Problem?
2. Wie funktioniert DAgger?
3. Was ist Goal-Conditioned Behavioral Cloning?
4. Warum funktioniert Behavioral Cloning manchmal trotzdem?
