# ZUSAMMENFASSUNG 06: Imitation Learning

## Übersicht
- **Seitenzahl PDF:** ~40 Seiten
- **Hauptthemen:** Behavioral Cloning, Distributional Shift, DAgger, Goal-Conditioned Imitation Learning
- **Klausurrelevanz:** ✅ WICHTIG (Grundkonzept IL, DAgger, Distributional Shift)
- **Lernzeit empfohlen:** 1-1.5h

---

## Detaillierte Inhalte

### 1. Behavioral Cloning (Grundkonzept)

#### Idee
- **Lerne Policy direkt von Experten-Demonstrationen** durch Supervised Learning
- Input: Zustand/Beobachtung $o_t$
- Target: Experten-Aktion $a_t$
- **Eines der ersten Deep Imitation Learning Systeme:** ALVINN (1989) - Autonomous Land Vehicle In a Neural Network

#### Mathematische Formulierung
**Policy:** $\pi_\theta(a|o)$ soll Experten-Policy $\pi^*(a|o)$ approximieren

**Training (Maximum Likelihood):**
$$\max_\theta \mathbb{E}_{(o,a) \sim \mathcal{D}_{demo}}[\log \pi_\theta(a|o)]$$

**Loss-Funktion (Negative Log-Likelihood):**
$$\mathcal{L} = -\mathbb{E}_{(o,a) \sim \mathcal{D}_{demo}}[\log \pi_\theta(a|o)]$$

#### Intuition
- Einfaches Supervised Learning auf (Zustand, Aktion)-Paaren
- Netzwerk lernt: "In Situation X macht der Experte Aktion Y"
- **Anwendung:** Autonomes Fahren, Robotik, Spiel-Agenten

---

### 2. Funktioniert Behavioral Cloning?

#### Das zentrale Problem: Fehlerakkumulation

**Beobachtung:** Gelernte Policy ist nicht perfekt – macht zumindest kleine Fehler

**Fehler-Kaskade:**
1. Policy macht kleinen Fehler
2. Fehler bringt Agenten in **Zustände, die nicht in den Trainingsdaten enthalten sind**
3. In diesen neuen Zuständen ist die Policy unsicher
4. Dadurch werden weitere Fehler mit jedem Schritt immer größer
5. **Fehler akkumulieren sich exponentiell!**

#### Beispiel: Autonomes Fahren
- **Training:** Experte fährt immer perfekt in der Spur
- **Test:** Policy macht kleinen Lenkfehler → Auto driftet zur Spurmitte
- **Problem:** Trainingsdaten enthalten nur "perfekt in Spur"-Zustände
- **Folge:** Policy weiß nicht, wie sie zurück in die Spur lenkt
- **Resultat:** Auto verlässt die Spur komplett

---

### 3. Distributional Shift - Formale Analyse

#### Trainings- vs. Test-Verteilung

**Training:** Daten werden unter Experten-Policy gesammelt
$$p_{train}(o_t) = p_{\pi^*}(o_t)$$

**Test:** Policy wird unter eigener Policy ausgeführt
$$p_{test}(o_t) = p_{\pi_\theta}(o_t)$$

**Problem:** 
$$p_{train}(o_t) \neq p_{test}(o_t)$$

#### Zielfunktion definieren ("was wir wollen")

**Cost-Funktion:**
$$c(s_t, a_t) = \begin{cases} 
0 & \text{wenn } a_t = \pi^*(s_t) \\
1 & \text{sonst}
\end{cases}$$

**Ziel:** Minimiere $\mathbb{E}_{\tau \sim p_{\pi_\theta}}[c(s_t, a_t)]$

Minimiere Anzahl der Fehler, die die Policy macht.

#### Worst-Case Analyse

**Annahme (Supervised Learning):**
$$\pi_\theta(a \neq \pi^*(s)|s) \leq \epsilon \quad \forall s \in \mathcal{D}_{train}$$

**Fehler ab dem ersten Zeitschritt:** $\approx 1$ (wenn $\epsilon \ll 1$)

**Für Zustände außerhalb der Trainingsdaten:**
$$\mathbb{E}_\tau[c(s_t, a_t)] = O(\epsilon T)$$

**Aber mit Distributional Shift:**
$$\mathbb{E}_\tau\left[\sum_{t=1}^T c(s_t, a_t)\right] \leq \epsilon T + (1-\epsilon)\epsilon(T-1) + (1-\epsilon)^2\epsilon(T-2) + \dots$$

**Resultat:** $O(\epsilon T^2)$ statt $O(\epsilon T)$!

**Interpretation:**
- **Ohne Distributional Shift (i.i.d.):** Lineare Fehlerakkumulation $O(\epsilon T)$
- **Mit Distributional Shift:** Quadratische Fehlerakkumulation $O(\epsilon T^2)$
- **T Terme, jeder $O(\epsilon T)$** → Fehler akkumulieren sich quadratisch!

#### Warum ist die Worst-Case Analyse zu pessimistisch?

In der Praxis können wir uns oft von Fehlern erholen:
- Imitation Learning hat aber **keinen eingebauten Mechanismus**, der das garantiert
- **Paradoxon:** Imitation Learning kann besser funktionieren, wenn Daten mehr Fehler (und dadurch Korrekturen) enthalten

---

### 4. Warum hat NVIDIA DRIVES (2016) funktioniert?

#### Erfolgsbeispiel: Bojarski et al. 2016 (NVIDIA)

**Beobachtung:** Behavioral Cloning funktionierte in der Praxis überraschend gut!

**Zitat aus dem Paper:**
> "Training with data from only the human driver is not sufficient. The network must learn how to recover from mistakes. Otherwise the car will slowly drift off the road. The training data is therefore augmented with additional images that show the car in different shifts from the center of the lane and rotations from the direction of the road."

#### Lösung: Data Augmentation mit seitlichen Kameras

**Technik:**
- Auto hat **3 Kameras**: mitte, links, rechts
- **Mitte-Kamera:** Normale Experten-Daten (Auto in Spurmitte)
- **Links-Kamera:** Simuliert Auto driftet nach rechts → Experte lenkt nach links
- **Rechts-Kamera:** Simuliert Auto driftet nach links → Experte lenkt nach rechts

**Effekt:**
- Trainingsdaten enthalten **Fehler + Korrekturen**
- Policy lernt nicht nur "in Spur bleiben", sondern auch "zurück in Spur lenken"
- **Breite der Trainingsdaten** deckt mehr Zustände ab

---

### 5. Learnings aus der Theorie

#### Kernaussagen

1. **Imitation Learning mittels Behavioral Cloning hat keine Garantien**
   - Anders als klassisches Supervised Learning
   - Grund: i.i.d. Annahmen gelten nicht

2. **Man kann die Gründe formalisieren (Theorie)**
   - Distributional Shift zwischen Trainings- und Test-Verteilung
   - Quadratische Fehlerakkumulation

3. **Mögliche Lösungen des Problems:**
   - Daten auf intelligente Art sammeln (und augmentieren)
   - Mächtige Modelle nutzen, die wenige Fehler machen
   - Multi-Task-Learning
   - Algorithmus anpassen (z.B. DAgger)

---

### 6. Was macht Behavioral Cloning leicht und was macht es schwer?

#### Strategien zur Verbesserung

##### A) Bewusst Fehler und Korrekturen bereitstellen

**Prinzip:** Fehler schaden, aber Korrekturen helfen oft mehr als Fehler schaden

**Umsetzung:**
- Data Augmentation mit simulierten Fehlern
- "Fake" Data, die Korrekturen demonstrieren (z.B. seitwärts gerichtete Kameras)
- Fehler unkorreliert mit Zustand mitteln sich beim Training teilweise aus

##### B) Mächtige Modelle nutzen

**Prinzip:** Deep Learning Modelle machen weniger Fehler → bessere Generalisierung

**Umsetzung:**
- Größere Netzwerke
- Bessere Architekturen (z.B. CNNs für Bilder)
- Mehr Trainingsdaten

##### C) Multi-Task-Learning

**Prinzip:** Policy lernt mehrere Tasks gleichzeitig → bessere Generalisierung

**Umsetzung:**
- Goal-Conditioned Behavioral Cloning (siehe unten)
- Policy $\pi(a|s, g)$ für beliebige Ziele $g$
- Mehr verschiedene Zustände in Trainingsdaten

---

### 7. Warum macht unsere Policy Fehler?

#### Grund 1: Experten-Verhalten ist nicht Markov

**Problem:**
- **Behavioral Cloning:** $\pi_\theta(a_t|o_t)$ - Verhalten hängt nur von aktueller Beobachtung ab
- **Experte:** $\pi^*(a_t|o_1, \dots, o_t)$ - Verhalten hängt von всей History ab

**Beispiel:**
- Auto bremst vor Kurve
- Policy sieht nur aktuellen Frame → weiß nicht warum
- Experte hat vorherige Frames gesehen → weiß dass Kurve kommt

**Konsequenz:** Wenn wir zweimal dasselbe sehen, machen wir zweimal dasselbe, egal was davor war. Oft sehr unnatürlich für Menschen.

#### Lösung: RNN/LSTM für History

**Variable Anzahl an Beobachtungen:**
$$\pi_\theta(a_t|o_1, \dots, o_t)$$

**Implementierung:**
- RNN/LSTM verarbeitet gesamte History
- Hidden State kodiert Vergangenheit
- Policy basiert auf Hidden State

**Problem:** 
1. Variable Anzahl an Beobachtungen
2. Zu viele Gewichte bei langen Sequenzen

**Praktische Lösung:** Begrenzte History (z.B. letzte 10 Frames) oder Attention-Mechanismen

#### Causal Confusion

**Problem:** Policy lernt falsche Kausalitäten

**Beispiel (de Haan et al., "Causal Confusion in Imitation Learning"):**
- Experte bremst wenn rotes Licht
- Policy lernt: "bremsen wenn rotes Pixel im Bild"
- Policy versteht nicht dass rotes Licht die **Ursache** ist
- **Folge:** Policy bremst bei allem Roten (Autos, Kleidung, etc.)

---

#### Grund 2: Multimodales Verhalten

**Problem:** Mehrere korrekte Aktionen möglich

**Beispiel:**
- Auto kann Spur halten durch: leicht nach links, geradeaus, leicht nach rechts lenken
- Alle drei Aktionen sind korrekt
- **Mittlerer Wert:** genau geradeaus → aber das ist nicht robust

#### Problem bei einfacher Wahrscheinlichkeitsverteilung

**Gaussian Policy:**
$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma^2)$$

**Problem:** Mittelwert von zwei guten Aktionen = schlechte Aktion

**Beispiel:** 
- Gute Aktion 1: stark nach links (-1)
- Gute Aktion 2: stark nach rechts (+1)
- **Mittelwert:** 0 (geradeaus) → Auto fährt in Gegenverkehr!

#### Lösungen für multimodales Verhalten

##### A) Ausdrucksstärkere kontinuierliche Wahrscheinlichkeitsverteilungen

**Mixture of Gaussians (MoG):**
$$\pi_\theta(a|s) = \sum_{k=1}^K w_k \mathcal{N}(\mu_k(s), \sigma_k^2)$$

**Vorteil:** Kann mehrere Modi abbilden

##### B) Diskretisierung von Aktionsräumen

**Diskrete Aktionen:**
- Links, Geradeaus, Rechts
- Kein "Mittelwert"-Problem

**Vorteil:** Kein Problem bei diskretem Aktionsraum

---

### 8. Goal-Conditioned Behavioral Cloning

#### Ist Multi-Task-Learning leichter?

**Standard Behavioral Cloning:**
$$\pi_\theta(a|s)$$
- Policy für **einen spezifischen Task**

**Goal-Conditioned Behavioral Cloning:**
$$\pi_\theta(a|s, g)$$
- Policy für **beliebige Ziele** $g$
- Viel mehr verschiedene Zustände → bessere Generalisierung

#### Training von Goal-Conditioned BC

**Daten:**
- Demo 1: $s_0^1, a_0^1, \dots, s_{T_1}^1, a_{T_1}^1$ (erfolgreiche Demo, um $g_1$ zu erreichen)
- Demo 2: $s_0^2, a_0^2, \dots, s_{T_2}^2, a_{T_2}^2$ (erfolgreiche Demo, um $g_2$ zu erreichen)
- Demo 3: $s_0^3, a_0^3, \dots, s_{T_3}^3, a_{T_3}^3$ (erfolgreiche Demo, um $g_3$ zu erreichen)

**Goal State:** $g = s_T$ (Endzustand der Demo)

**Loss-Funktion:**
Für jede Demo $i$ mit Trajektorie $(s_0^i, a_0^i, \dots, s_{T_i}^i, a_{T_i}^i)$ und Ziel $g_i = s_{T_i}^i$:

$$\max_\theta \sum_{i} \sum_{t=0}^{T_i-1} \log \pi_\theta(a_t^i | s_t^i, g_i)$$

**Intuition:**
- Policy lernt: "In Zustand $s$, um Ziel $g$ zu erreichen, mache Aktion $a$"
- Generalisiert auf neue Ziele
- Robuster gegenüber Distributional Shift

---

### 9. Beispiele für Goal-Conditioned IL

#### Learning Latent Plans from Play

**Paper:** Lynch et al. 2020
**URL:** https://arxiv.org/pdf/1903.01973

**Idee:**
1. **Daten sammeln:** Roboter interagiert zufällig mit Umgebung ("play")
2. **Latent Plans:** Extrahiere implizite Ziele aus Trajektorien
3. **Training:** Goal-conditioned Policy auf gesammelten Daten
4. **Execution:** Policy erreicht neue Ziele durch Kombination gelernter Fähigkeiten

**Vorteil:** Keine expliziten Labels nötig, unsupervised Datensammlung

---

#### Unsupervised Visuomotor Control through Distributional Planning Networks

**Idee:** Visuelle Planung ohne explizite Labels

**Ansatz:**
- Lerne visuelle Repräsentationen aus Rohbildern
- Plane in latentem Raum
- Goal-conditioned Policy erreicht visuelle Ziele

---

### 10. Automatisierte Datensammlung

#### Learning to Reach Goals via Iterated Supervised Learning

**Paper:** Ghosh et al. 2019
**URL:** https://arxiv.org/abs/1912.06088

**Algorithmus:**

1. **Starte mit Zufallspolicy** $\pi_0$
2. **Sammle Daten** mit zufälligen Zielen unter aktueller Policy
3. **Behavioral Cloning** auf den Zielen, die tatsächlich erreicht wurden
4. **Nutze dies um die Policy zu verbessern** → $\pi_{k+1}$
5. **Wiederhole** ab Schritt 2

**Intuition:**
- Policy wird schrittweise besser
- Jede Iteration sammelt Daten in schwereren Zuständen
- **Automatisiert:** Kein menschlicher Experte nötig!

**Vorteile:**
- Unbegrenzte Menge an Daten aus eigener Erfahrung
- Kontinuierliche Selbstverbesserung
- Keine menschlichen Demonstrationen nötig

---

### 11. DAgger (Dataset Aggregation)

#### Motivation

**Problem:** $p_{\pi_\theta}(o_t) \neq p_{\pi^*}(o_t)$

**Frage:** Können wir bewerkstelligen, dass:
$$p_{train}(o_t) = p_{\pi_\theta}(o_t)$$

**Idee:** Anstatt $p_{\pi_\theta}(o_t)$ anzupassen, versuche $p_{train}(o_t)$ anzupassen. D.h., $p_{train}(o_t)$ soll einen größeren Bereich an Zuständen abdecken, die in $p_{\pi_\theta}(o_t)$ enthalten sind (also von $\pi_\theta(a_t|o_t)$ besucht werden).

#### DAgger Algorithmus

**Quelle:** Ross et al. PMLR 15:627-635, 2011

**Algorithmus:**

1. **Trainiere** $\pi_\theta(a_t|o_t)$ auf Expertendaten $\mathcal{D} = \{(o_1, a_1), \dots, (o_N, a_N)\}$

2. **Führe** $\pi_\theta(a_t|o_t)$ aus, erhalte Datensatz $\mathcal{D}_\pi = \{o_1, \dots, o_M\}$

3. **Experte labelt** $\mathcal{D}_\pi$ mit korrekten Aktionen $a_t^*$

4. **Aggregiere** $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_\pi$

5. **Wiederhole** ab Schritt 1 mit aggregiertem Dataset

#### Warum DAgger funktioniert

**Prinzip:**
- Neue Daten enthalten Zustände, die die Policy **tatsächlich besucht**
- Experte labelt diese Zustände mit korrekten Aktionen
- **Trainingsdaten decken Test-Verteilung ab**

**Theoretische Garantie:**
$$\mathbb{E}\left[\sum_{t=1}^T c(s_t, a_t)\right] = O(T)$$

Statt $O(T^2)$ bei normalem Behavioral Cloning!

#### DAgger Schwachstellen

1. **Kann aufwendig sein:** Experte muss viele Zustände labeln

2. **Kann "unnatürlich" sein:** Daten im Nachhinein zu labeln führt zu anderen Aktionen als der Experte im Live-Betrieb gewählt hätte

3. **Experte muss online verfügbar sein:** Nicht immer praktikabel

**Praktische Lösung:**
- Experte labelt nur kritische Zustände
- Automatische Labeling-Heuristiken für einfache Fälle

---

### 12. Zusammenfassung: Lösungsansätze für Distributional Shift

| Ansatz | Idee | Vorteil | Nachteil |
|--------|------|---------|----------|
| **Data Augmentation** | Fehler + Korrekturen simulieren | Einfach, effektiv | Benötigt Domänenwissen |
| **Mächtige Modelle** | Weniger Fehler durch bessere Architektur | Generalisierung | Mehr Rechenleistung |
| **Multi-Task-Learning** | Goal-conditioned Policy | Breite Zustandsabdeckung | Komplexeres Training |
| **DAgger** | Iterative Datensammlung | Theoretische Garantie | Experte muss online labeln |
| **Automatisierte Datensammlung** | Self-supervised Learning | Kein Experte nötig | Langsamer Lernprozess |

---

### 13. Warum ist Imitation Learning nicht genug?

#### Limitationen von Imitation Learning

1. **Menschen müssen Daten bereitstellen** → begrenzte Menge an Daten
   - Deep Learning funktioniert umso besser, je mehr Daten zur Verfügung stehen

2. **Menschen können nicht für alles passende Aktionen bereitstellen**
   - Neue Situationen, die der Experte nicht kennt

3. **Menschen können autonom lernen** → können Maschinen das auch?

#### Vorteile von Reinforcement Learning

1. **Unbegrenzte Menge an Daten** aus eigener Erfahrung

2. **Kontinuierliche Selbstverbesserung** durch Exploration

3. **Entdeckung neuer Strategien** die besser als Experten sind

**Aber:**
- RL benötigt viele Interaktionen mit Umgebung
- Imitation Learning ist sample-effizienter
- **Kombination:** IL für Initialisierung, RL für Feinabstimmung

---

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ Distributional Shift Problem
- **Warum:** Kernproblem von Behavioral Cloning
- **Was:** $p_{train} \neq p_{test}$, Fehler-Akkumulation $O(\epsilon T^2)$
- **Klausurfrage:** "Erklären Sie warum Behavioral Cloning scheitern kann"

### ✅ DAgger Algorithmus
- **Warum:** Wichtige Lösung für Distributional Shift
- **Was:** Dataset Aggregation, iterative Verbesserung
- **Garantie:** Lineare Fehlerakkumulation $O(T)$ statt quadratisch
- **Klausurfrage:** "Wie funktioniert DAgger und warum löst es das Problem?"

### ✅ Goal-Conditioned IL
- **Warum:** Moderne Alternative, Multi-Task Ansatz
- **Was:** Policy $\pi(a|s, g)$ für beliebige Ziele
- **Vorteil:** Bessere Generalisierung
- **Klausurfrage:** "Was ist Goal-Conditioned Behavioral Cloning?"

### ✅ NVIDIA DRIVES Beispiel
- **Warum:** Praxisbeispiel für erfolgreiches IL
- **Was:** Data Augmentation mit seitlichen Kameras
- **Prinzip:** Fehler + Korrekturen in Trainingsdaten

---

## Formeln/Algorithmen (wichtig)

### Behavioral Cloning Loss
$$\mathcal{L} = -\mathbb{E}_{(o,a) \sim \mathcal{D}_{demo}}[\log \pi_\theta(a|o)]$$

### Distributional Shift (Worst-Case)
$$\mathbb{E}\left[\sum_{t=1}^T c(s_t, a_t)\right] = O(\epsilon T^2)$$

### DAgger Algorithmus
```
for iteration i = 1, 2, ...:
    D_i = rollout(π_i)              # Ausführen der aktuellen Policy
    D_i^labeled = expert_label(D_i)  # Experte labelt Zustände
    D = D ∪ D_i^labeled              # Dataset aggregieren
    π_{i+1} = train(D)               # Neue Policy trainieren
```

### Goal-Conditioned BC Loss
$$\mathcal{L} = -\sum_{i} \sum_{t=0}^{T_i-1} \log \pi_\theta(a_t^i | s_t^i, g_i=s_{T_i}^i)$$

### UCB (für Exploration in RL)
$$a = \arg\max_a \left[\hat{\mu}_a + \sqrt{\frac{2 \ln(T)}{N(a)}}\right]$$

---

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte
- **Behavioral Cloning:** Einfach aber Distributional Shift Problem
- **Distributional Shift:** $p_{train} \neq p_{test}$ → quadratische Fehlerakkumulation
- **DAgger:** Löst Problem durch iterative Datensammlung mit Experten-Labels
- **Goal-Conditioned:** Multi-Task Ansatz für bessere Generalisierung
- **Data Augmentation:** Fehler + Korrekturen in Trainingsdaten (NVIDIA Beispiel)

### ⚠️ Häufige Fehler
- Nicht erkennen, dass i.i.d. Annahme verletzt ist
- DAgger vs Behavioral Cloning verwechseln
- Denken dass IL Garantien hat (hat es nicht!)
- Multimodales Verhalten mit Gaussian Policy modellieren (führt zu Mittelwert-Problem)

### 📝 Prüfungsrelevante Fragen
1. Was ist Distributional Shift und warum ist es ein Problem?
2. Wie funktioniert DAgger und welche Garantie bietet es?
3. Was ist Goal-Conditioned Behavioral Cloning?
4. Warum funktioniert Behavioral Cloning manchmal trotzdem (NVIDIA Beispiel)?
5. Was ist das multimodale Verhaltensproblem und wie löst man es?
6. Unterschied zwischen On-Policy und Off-Policy bei IL?
7. Warum ist Experten-Verhalten oft nicht Markov?

### 🔗 Querverweise
- **RL-Teil:** Q-Learning, Policy Gradients (Alternative zu IL)
- **Transformers:** Attention-Mechanismus (kann für History in IL genutzt werden)
- **XAI:** Warum hat Policy falsch entschieden? (Causal Confusion)

---

## Tipps für die Klausur

✅ **Prioritäten:**
- Distributional Shift Konzept verstehen (wichtigste!)
- DAgger Algorithmus skizzieren können
- Goal-Conditioned IL Grundidee kennen

✅ **Typische Frage-Typen:**
- "Erklären Sie..." - Distributional Shift, DAgger
- "Was ist der Unterschied..." - BC vs DAgger, IL vs RL
- "Wie funktioniert..." - DAgger Algorithmus, Goal-Conditioned BC

✅ **Wichtige Konzepte (auswendig):**
- Distributional Shift: $p_{train} \neq p_{test}$
- DAgger: Iterative Datensammlung mit Experten-Labels
- Goal-Conditioned: $\pi(a|s, g)$ statt $\pi(a|s)$

---

**Erstellt:** 17.03.2026  
**Aktualisiert:** 17.03.2026 (sehr ausführlich erweitert)  
**Basierend auf:** AdvancedML-06-IL.pdf (~40 Seiten)
