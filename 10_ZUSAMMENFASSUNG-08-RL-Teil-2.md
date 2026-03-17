# ZUSAMMENFASSUNG 08: Reinforcement Learning Teil 2

## Übersicht
- **Seitenzahl PDF:** ~76 Seiten (AdvancedML-08-RL-Teil-2.pdf)
- **Hauptthemen:** Exploration in Deep RL, Offline Reinforcement Learning
- **Klausurrelevanz:** 🔴 SEHR WICHTIG - UCB, Thompson Sampling, Offline RL Problem, DAgger vs. Offline RL
- **Lernzeit empfohlen:** 2-2.5h

---

## Detaillierte Inhalte

### 1. Einführung: Warum ist Exploration schwierig?

#### Schwierigkeitsgrad aus Sicht eines RL-Agenten

**Extrem schwierige Aufgabe:** Montezuma's Revenge
- Schlüssel aufsammeln → Reward
- Tür öffnen → Reward
- Getötet werden von Totenkopf → nichts (gut? schlecht?)
- **Problem:** Das Spiel zu beenden korreliert nur schwach mit Reward-Ereignissen
- Menschliche Spieler wissen was zu tun ist, weil sie die Spielelemente verstehen (Schlüssel, Totenkopf, etc.)

**Eher leichte Aufgabe:** Breakout
- Klare Korrelation zwischen Aktion und Reward
- Unmittelbares Feedback

#### Warum verstehen RL-Agenten die Regeln nicht?

**Sicht eines RL-Algorithmus:**
- Sie **kennen die Regeln nicht**
- Können Regeln **nur durch Ausprobieren** herausfinden
- Regeln machen evtl. **keinen Sinn** aus ihrer Sicht
- **Zeitlich ausgedehnte Aufgaben** wie Montezuma's Revenge werden umso schwieriger

**Beispiel UNO:**
- Je größer der Zeithorizont ist
- Wie wenig Sie über die Regeln wissen
- **Stellen Sie sich vor, Ihr Lebensziel bestünde daraus, 42 Spiele UNO zu gewinnen...**
- **...und Sie wüssten vorher nichts davon**

#### Weitere Beispiele außerhalb von Spielen

Das beschriebene Problem beschränkt sich nicht nur auf Spiele:
- **Robotik:**
  - Gegenstände mit einer (Roboter-)Hand bewegen
  - Werkzeuge nutzen
  - Türen oder Schubladen öffnen
- **Erfordert komplexes Zusammenspiel mehrerer Finger** und das Erlernen, wie auf die jeweilige Umgebung Einfluss genommen werden kann

---

### 2. Exploration vs. Exploitation

#### Das fundamentale Dilemma

**Frage:** Wie kann ein Agent ein Verhalten erlernen, das eine zeitlich ausgedehnte Sequenz komplexer Aktionen erfordert, die individuell keinen Reward bringen?

**Entscheidung:** Wie kann ein Agent sich entscheiden, ob:
- **Neues Verhalten probiert** (mit dem Ziel, höhere Rewards zu entdecken) **ODER**
- **Weiterhin das bestbekannte Verhalten** fortgesetzt werden soll

#### Definitionen

| Begriff | Definition | Ziel |
|---------|------------|------|
| **Exploitation** | Tun, was wir aktuell für das beste halten | Kurzfristiger Gewinn |
| **Exploration** | Neue Dinge ausprobieren in der Hoffnung noch höhere Rewards zu erhalten | Langfristiger Gewinn |

**Trade-off:** Zu viel Exploration → schlechte Performance. Zu viel Exploitation → suboptimale Policy wird gelernt.

---

### 3. Theoretische Grundlagen der Exploration

#### Wie können wir eine optimale Explorationsstrategie finden?

**Frage:** Was bedeutet "optimal" in dem Kontext?
- **Antwort:** Minimierung des Regret im Vergleich zu einer Bayes-optimalen Strategie

#### Schwierigkeitsklassen

| Problemklasse | Lösbarkeit |
|---------------|------------|
| **Multi-armed Bandits** (1-step stateless RL problems) | Theoretisch lösbar |
| **Contextual Bandits** (1-step RL problems) | Theoretisch lösbar |
| **Kleine, endliche MDPs** (z.B. berechenbare Planung, modellbasiertes RL) | Theoretisch lösbar |
| **Große, unendliche MDPs, kontinuierliche Räume** | Theoretisch unlösbar |

---

### 4. Bandits - Das einfachste Explorationsproblem

#### Was sind Banditen?

**One-armed Bandit:**
- $\mathcal{A} = \{\text{Arm ziehen}\}$
- $r(a) = ?$ (unbekannt)
- **Einfachstes aller Explorationsprobleme**

**Multi-armed Bandit:**
- $\mathcal{A} = \{Arm_1, Arm_2, \dots, Arm_K\}$
- $r(a_i) \sim p(r|a_i)$ (jede Aktion hat eigene Reward-Verteilung)

#### Formale Definition

**Annahme:** $r(a_t) \sim p_\theta(r_t)$
- z.B. $p(r_t = 1) = \theta_a$ und $p(r_t = 0) = 1 - \theta_a$ (Bernoulli-Verteilung)

**Dies definiert einen POMDP mit:**
- $s = (\theta_1, \dots, \theta_K)$ (latente Parameter)
- **"belief state"** $\hat{p}(\theta_1, \dots, \theta_K)$ (unsere Schätzung der Parameter)

**Lösung des POMDP** → optimale Explorationsstrategie:
- "Welche Aktion ist jetzt optimal, wenn ich auch berücksichtige, was ich dadurch lerne und wie mir dieses Wissen in allen künftigen Schritten hilft?"

**Problem:** Das ist ein riesiges, nicht allgemein lösbares Problem!

**Aber:** Wir können sehr gute Ergebnisse mit einfachen Strategien erzielen.

#### Regret - Qualitätsmaß für Exploration

**Definition:**
$$\text{Reg}(T) = T \cdot \mathbb{E}[r(a^*)] - \sum_{t=1}^T \mathbb{E}[r(a_t)]$$

**Komponenten:**
- $T \cdot \mathbb{E}[r(a^*)]$: Erwarteter Return der **optimalen Policy** (beste Aktion in jedem Schritt)
- $\sum_{t=1}^T \mathbb{E}[r(a_t)]$: Tatsächlich durchgeführte Aktionen

**Interpretation:**
- Regret misst die **Differenz zur optimalen Policy** zum Zeitpunkt $T$
- **Ziel:** Regret minimieren
- **Optimale Strategien** haben $\text{Reg}(T) = O(\log T)$ (beweisbar optimal für Bandits)

---

### 5. Drei Explorationsstrategie-Klassen

#### Klasse 1: Optimistische Exploration (UCB)

**Grundidee:** "Optimismus bei Unsicherheit"

**Algorithmus:**
1. Tracke durchschnittlichen Reward $\hat{\mu}_a$ für jede Aktion $a$
2. **Exploitation:** wähle $a = \arg\max_a \hat{\mu}_a$
3. **Optimistische Schätzung:** $a = \arg\max_a [\hat{\mu}_a + C \cdot \sigma_a]$

**Spezifische Form (Auer et al. 2002):**
$$a = \arg\max_a \left[\hat{\mu}_a + \sqrt{\frac{2 \ln(T)}{N(a)}}\right]$$

**Komponenten:**
- $\hat{\mu}_a$: Geschätzter durchschnittlicher Reward für Aktion $a$
- $N(a)$: Anzahl der Besuche von Aktion $a$
- $T$: Gesamtzahl der Schritte
- $\sqrt{\frac{2 \ln(T)}{N(a)}}$: **Explorations-Bonus** (Unsicherheit)

**Intuition:**
- Versuche jeden Arm bis es sicher ist, dass er nicht gut ist
- **Bei Unsicherheit → Optimismus**
- Bonus sinkt mit $N(a)$: Mehr Besuche → weniger Unsicherheit
- Bonus steigt mit $\ln(T)$: Späte Zeitpunkte → höhere Exploration für unbesuchte Aktionen

**Theoretische Garantie:**
$$\text{Reg}(T) = O(\log T)$$
**Beweisbar optimal** (im Bandit-Setup)!

---

#### Klasse 2: Thompson Sampling

**Grundidee:** "Sample aus Posterior, handle optimal unter Sample"

**Annahme:** $r(a_t) \sim p_\theta(r_t)$ definiert POMDP mit $s = (\theta_1, \dots, \theta_K)$

**"belief state":** $\hat{p}(\theta_1, \dots, \theta_K)$ (Modell des Bandits)

#### Thompson Sampling Algorithmus

```
1. Sample θ₁, ..., θ_K ~ p̂(θ₁, ..., θ_K)    # Aus Posterior-Verteilung
2. Nehme an, θ₁, ..., θ_K seien korrekt
3. Nehme optimale Aktion: a = argmax_a E_θ[r|a]
4. Update Modell aus Beobachtung
5. Wiederhole ab Schritt 1
```

**Alternative Namen:**
- Posterior Sampling
- Probability Matching

**Eigenschaften:**
- **Schwer theoretisch zu analysieren** (aber gute Regret-Bounds bewiesen)
- **Funktioniert gut empirisch** (oft besser als UCB in der Praxis)
- **Bayesianischer Ansatz:** Maintain Posterior-Verteilung über Parameter

**Implementation (Beta-Binomial für Bernoulli-Bandits):**
- **Prior:** $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$
- **Update bei Erfolg (r=1):** $\alpha_a \leftarrow \alpha_a + 1$
- **Update bei Misserfolg (r=0):** $\beta_a \leftarrow \beta_a + 1$
- **Sample:** $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$

---

#### Klasse 3: Information Gain

**Grundidee:** "Wähle Aktion mit maximalem Informationsgewinn"

**Definitionen:**

**Latente Variable $z$:**
- $z$ könnte z.B. die optimale Aktion oder der Wert einer Aktion sein

**Entropie:**
- $H(\hat{p}(z))$: Entropie unserer aktuellen Schätzung von $z$
- $H(\hat{p}(z|y))$: Entropie unserer Schätzung von $z$ nach Beobachtung $y$
- **Je geringer die Entropie, desto präziser kennen wir $z$**

**Information Gain (IG):**
$$IG(z, y) = \mathbb{E}_y[H(\hat{p}(z)) - H(\hat{p}(z|y))]$$

**Aktionsabhängig:**
$$IG(z, y|a) = \mathbb{E}_y[H(\hat{p}(z)) - H(\hat{p}(z|y)) | a]$$

**Interpretation:**
- Wie viel lernen wir über $z$ aus Aktion $a$, gegeben unsere aktuellen Beliefs?

#### Information-Directed Sampling (Russo & Van Roy)

**Beispiel eines Bandit-Algorithmus:**
- $y = r_a$ (Reward von Aktion $a$)
- $z = \theta_a$ (Parameter des Modells $p(r_a)$)
- $g(a) = IG(\theta_a, r_a | a)$: Information Gain von Aktion $a$
- $\Delta(a) = \mathbb{E}[r(a^*) - r(a)]$: Erwartete Suboptimalität von $a$

**Auswahlregel:**
$$a = \arg\min_a \frac{\Delta(a)^2}{g(a)}$$

**Intuition:**
- **Wähle keine sicher suboptimalen Aktionen** ($\Delta(a)$ groß)
- **Wähle keine Aktionen, aus denen du nichts lernst** ($g(a)$ klein)
- **Balance:** Aktionen mit gutem Verhältnis von Suboptimalität zu Information Gain

---

### 6. Zusammenfassung: Leitideen der Exploration

| Methode | Formel | Intuition |
|---------|--------|-----------|
| **UCB** | $a = \arg\max_a [\hat{\mu}_a + \sqrt{\frac{2 \ln(T)}{N(a)}}]$ | Optimismus bei Unsicherheit |
| **Thompson Sampling** | $\theta_1, \dots, \theta_K \sim \hat{p}(\theta)$, $a = \arg\max_a \mathbb{E}_{\theta_a}[r|a]$ | Sample aus Posterior, handle optimal |
| **Information Gain** | $IG(z, y|a)$ | Maximiere Informationsgewinn |

**Gemeinsame Prinzipien:**
1. **Die meisten Explorationsstrategien benötigen eine Art Unsicherheitsabschätzung** (auch wenn sie sehr grob ist)
2. **Üblicherweise wird angenommen, dass neue Information einen Wert hat:**
   - Unbekanntes = potenziell gut (Optimismus)
   - Stichprobe = Wahrheit
   - Informationsgewinn = gut

---

### 7. Exploration in Deep RL

#### Übertragung auf Deep RL

**Warum sollten wir uns dafür interessieren?**
- Bandits sind leichter zu analysieren und zu verstehen
- Sie liefern grundlegende Einsichten für Explorationsmethoden
- Diese Methoden lassen sich anschließend auf komplexere MDPs anwenden

#### Verschiedene Explorations-Methoden in Deep RL

| Methode | Idee | Implementierung |
|---------|------|-----------------|
| **Optimistische Exploration** | Neuer Zustand = guter Zustand | Erfordert Zustandscounter oder Neuigkeit von Zuständen messen, typischerweise realisiert über Explorationsbonus |
| **Thompson Sampling** | Lerne Verteilung über Q-Functions oder Policies | Sample und agiere gemäß Sample |
| **Information Gain** | Erwäge Information Gain durch Besuch neuer Zustände | Schätzung des IG durch Modelländerung |

---

### 8. Optimistische Exploration in Deep RL

#### Grundprinzip

**UCB-Idee übertragen auf RL:**
$$a = \arg\max_a \left[\mu_a + \frac{2 \ln T}{N(a)}\right]$$

**"Explorationsbonus":**
- Viele Funktionen möglich, solange sie mit $N(a)$ abnehmen

**Count-basierte Exploration:**
- Verwende $N(s, a)$ oder $N(s)$ für Explorationsbonus
- **Modifizierter Reward:**
  $$r^+(s, a) = r(s, a) + B(N(s))$$
- $B(N(s))$: Bonus, der mit $N(s)$ abnimmt
- **Nutze $r^+(s, a)$ statt $r(s, a)$** mit beliebigem modellfreien Algorithmus

**Vorteile:**
- Einfache Ergänzung für jeden RL-Algorithmus

**Nachteil:**
- Tuning von Bonusgewichten notwendig

---

### 9. Pseudo-Counts - Zählen in großen Zustandsräumen

#### Das Problem

**Frage:** Wie zählen wir Zustände in riesigen Zustandsräumen?
- Viele Zustände sehen wir nur ein einziges Mal!
- Aber einige Zustände ähneln einander mehr als andere...

#### Idee: Modelle fitten

**Ansatz:** Fitte Modell für Zustandsdichte $p_\theta(s)$ oder $p_\theta(s, a)$ als Näherung für $N(s)$ bzw. $N(s, a)$

**Interpretation:**
- $p_\theta(s)$ hoch → $s$ ist ähnlich zu einem zuvor schon gesehenen Zustand
- Wir können $p_\theta(s)$ als **"Pseudo-Count"** nutzen

#### Mathematische Herleitung

**Für kleine MDPs:**
- Nach Beobachtung von $s$: $p_\theta(s) = \frac{n}{N}$
- Nach nochmaliger Beobachtung: $p_{\theta'}(s) = \frac{n+1}{N+1}$

**Idee:** Wenn $p_\theta(s)$ und $p_{\theta'}(s)$ diesen Gleichungen gehorchen, können wir daraus $N(s)$ und $n$ bestimmen.

#### Algorithmus: Exploration mit Pseudo-Counts

```
1. Fitte Modell p_θ(s) an alle bisher gesehenen Zustände D
2. Mache einen Schritt i und beobachte s_i
3. Fitte neues Modell p_θ'(s) an D ∪ {s_i}
4. Nutze p_θ(s_i) und p_θ'(s_i) zur Schätzung von N(s_i)
5. r_i^+ = r_i + B(N̂(s_i))  # Pseudo-Count Bonus
```

#### Berechnung des Pseudo-Counts

**Gleichungssystem:**
$$p_\theta(s_i) = \frac{n}{N}$$
$$p_{\theta'}(s_i) = \frac{n+1}{N+1}$$

**Lösen nach $n$ und $N$:**
- Zwei Gleichungen mit zwei Unbekannten
- **Lösung:** $\hat{N}(s_i) = n \approx \frac{p_\theta(s_i)}{p_{\theta'}(s_i) - p_\theta(s_i)}$

**Quelle:** Bellemare et al. "Unifying Count-Based Exploration..."

---

### 10. Bonus-Funktionen

Viele verschiedene Funktionen in der Literatur, inspiriert durch optimale Methoden für Bandits oder kleine MDPs:

| Bonus-Funktion | Formel | Quelle |
|----------------|--------|--------|
| **UCB** | $B(N(s)) = \sqrt{\frac{2 \ln n}{N(s)}}$ | Standard |
| **MBIE-EB** | $B(N(s)) = \frac{1}{N(s)}$ | Strehl & Littmann, 2008 |
| **BEB** | $B(N(s)) = \frac{1}{\sqrt{N(s)}}$ | Kolter & Ng, 2009 |

**Eigenschaften:**
- Alle Funktionen **nehmen mit $N(s)$ ab**
- Höherer Bonus für selten besuchte Zustände
- Tuning des Bonusgewichts oft notwendig

---

### 11. Weitere Neuigkeit-suchende Explorationsmethoden

#### Welches Modell nutzen?

**Anforderungen an $p_\theta(s)$:**
- Brauchen Zustandsdichten, aber müssen nicht unbedingt gute Samples produzieren können
- **Andere Anforderungen als bei generativen Modellen** (GANs, Diffusion)

**Beispiel-Modelle:**
- **CTS-Modell** (Bellemare et al.): Konditionierung jedes Pixels auf dessen Nachbarschaft links/oben
- **Stochastische neuronale Netzwerke**
- **Compression Length**
- **EX2** (Exemplar Models)

---

#### Zählen mit Hashes

**Idee:** Komprimiere $s$ in einen $k$-bit Code mit $\phi(s)$ und zähle $N(\phi(s))$

**Prinzip:**
- Kürzere Codes = mehr Hashkollisionen
- **Frage:** Erhalten ähnliche Zustände denselben Hash?
- **Antwort:** Kommt drauf an...

**Implementation:**
- Beeinflusse dies durch Lernen einer Kompression mit Autoencodern
- **Tang et al.:** "#Exploration: A Study of Count-Based Exploration"

---

#### Implizite Dichtemodellierung mit Exemplar-Modellen

**Intuition:** Der Zustand ist neu, wenn es leicht ist, ihn mittels Classifier von allen zuvor gesehenen Zuständen zu unterscheiden.

**Algorithmus:**
1. Für jeden Zustand $s$: Fitte Classifier $D_s$ um diesen Zustand gegen alle vorherigen Zustände $\mathcal{D}$ zu klassifizieren
2. Nutze dies, um eine Dichte zu erhalten:
   $$p_\theta(s) = \frac{1 - D_s(s)}{D_s(s)}$$
3. $D_s(s)$: Wahrscheinlichkeit, die der Classifier für "$s$ ist positiv" angibt

**Problem:** Task ist nicht trivial, da $s$ auch in $\mathcal{D}$ enthalten sein kann.

**Optimaler Classifier:**
- Wenn $s \in \mathcal{D}$, ist das optimale $D_s^*(s) \neq 1$, in der Tat:
  $$D_s^*(s) = \frac{1}{1 + |\mathcal{D}|}$$
  $$\Rightarrow p_\theta(s) = \frac{1 + |\mathcal{D}| - 1}{1} = |\mathcal{D}|$$

**In der Realität:**
- Jeder Zustand einzigartig (großer Zustandsraum)
- **Regularisierung gegen Overfitting** notwendig

**Praktische Implementation:**
- Ein Classifier pro Zustand ist aufwendig...
- **Amortisiertes Modell:** Einzelnes Netzwerk mit Exemplar als weiterer Input

**Quelle:** Fu et al. "EX2: Exploration with Exemplar Models..."

---

#### Heuristische Count-Schätzung via Fehler

**Idee:** Brauchen nicht unbedingt gute Dichten produzieren... müssen lediglich erkennen, ob ein Zustand neu ist oder nicht!

**Ansatz:**
1. Angenommen, wir haben eine Zielfunktion $f^*(s, a)$
2. Fitte $f_\phi(s, a)$ mit Daten aus unserem Buffer $\mathcal{D} = \{(s_i, a_i)\}$
3. Nutze $\mathcal{E}(s, a) = \|f_\phi(s, a) - f^*(s, a)\|^2$ als Bonus

**Welche Zielfunktion $f^*(s, a)$ wählen?**

**Option 1: Next State Prediction**
- $f^*(s, a) = s'$ (nächster Zustand)
- Modell lernt Transitionen vorherzusagen
- Hoher Vorhersagefehler = neuer/unbekannter Zustand

**Option 2: Random Network Distillation (einfacher!)**
- $f^*(s, a) = f_\phi(s, a)$ mit **zufälligem Parametervektor** $\phi$
- $f^*$ ist eine **beliebige feste Funktion**
- $f_\phi$ wird trainiert, $f^*$ zu approximieren
- **Hoher Fehler = neuer Zustand** (Netzwerk wurde noch nicht auf diesem Zustand trainiert)

**Quelle:** Burda et al. (2018), "Exploration by Random Network Distillation"

---

### 12. Thompson Sampling in Deep RL

#### Übertragung auf MDPs

**Thompson Sampling für Bandits:**
1. $\theta_1, \dots, \theta_K \sim \hat{p}(\theta_1, \dots, \theta_K)$
2. $a = \arg\max_a \mathbb{E}_{\theta_a}[r|a]$

**Fragen für MDPs:**
- Was wird gesampelt?
- Wie wird die Verteilung repräsentiert?

**MDP-Analogon:**
- Bandit Setting: $\hat{p}(\theta_1, \dots, \theta_K)$ ist Verteilung über Rewards
- **MDP-Analogon ist die Q-Funktion!**

#### Thompson Sampling für MDPs

**Algorithmus:**
1. Sample Q-Funktion $Q$ aus $p(Q)$
2. Handle gemäß $Q$ für eine Episode
3. Update $p(Q)$

**Herausforderung:** Wie können wir eine Verteilung über Funktionen ($p(Q)$) repräsentieren?

---

#### Bootstrapped DQN

**Idee:** Bootstrapping zur Approximation von $p(Q)$

**Algorithmus:**
1. Gegeben Datensatz $\mathcal{D}$, resample $N$ mal mit Zurücklegen → $\mathcal{D}_1, \dots, \mathcal{D}_N$
2. Trainiere je ein Modell $f_{\theta_i}$ auf $\mathcal{D}_i$
3. Sampeln aus $p(\theta)$ → sample $i \in [1, \dots, N]$ und nutze $f_{\theta_i}$

**Praktisches Problem:**
- Training großer neuronaler Netzwerke ist teuer
- **Lösung:** Alle Netzwerke auf demselben Datensatz trainieren, aber mit unterschiedlichen Bootstrap-Samples

**Quelle:** Osband et al. "Deep Exploration via Bootstrapped DQN"

---

#### Warum funktioniert Bootstrapped DQN?

**Vergleich:**

| Methode | Verhalten |
|---------|-----------|
| **Exploration mit Zufallsaktionen** (z.B. $\epsilon$-greedy) | Oszillieren hin und her, geht evtl. nicht zu interessanten Orten, **keine kohärente Policy** |
| **Exploration mit zufälligen Q-Funktionen** | Commitment zu einer zufälligen, aber **intern konsistenten Strategie für eine ganze Episode** |

**Vorteile:**
- Kohärente Exploration über ganze Episode
- Kein "Hin-und-Her-Oszillieren" wie bei $\epsilon$-greedy
- Agent probiert konsistente Strategien aus

**Nachteile:**
- Sehr gute Bonusse (Pseudo-Counts, etc.) funktionieren oft besser

---

### 13. Information Gain in Deep RL

#### Schätzung des Information Gain

**Information Gain:** $IG(z, y|a)$

**Frage:** Information Gain worüber?

| Option | Beschreibung | Nützlichkeit |
|--------|-------------|--------------|
| **IG über Reward $r(s, a)$** | Information über direkte Rewards | Nicht nützlich, wenn Rewards selten |
| **IG über Zustandsdichte $p(s)$** | Information über Zustandsverteilung | Schon besser, macht mehr Sinn |
| **IG über Dynamik $p(s'|s, a)$** | Information über Transitionen | Guter Proxy um MDP zu lernen |

**Problem:** Grundsätzlich nicht genau berechenbar, unabhängig davon, was geschätzt wird!

---

#### Näherungen für Information Gain

**1. Prediction Gain** (Schmidhuber '91, Bellemare '16)
$$\text{Bonus} = \log p_{\theta'}(s) - \log p_\theta(s)$$

**Intuition:** Wenn Dichte sich stark ändert, war der Zustand neu.

**2. Variational Inference (VIME)** (Houthooft et al.)

**Beobachtung:** IG kann geschrieben werden als KL-Divergenz:
$$IG = D_{KL}(p(\theta|z, y) \parallel p(\theta))$$

**Ansatz:**
- Lerne Transitions $p_\theta(s_{t+1}|s_t, a_t)$
- $z = \theta$ (Modellparameter), $y = (s_t, a_t, s_{t+1})$ (Transition)
- **Intuition:** Transition ist informativ, wenn es unseren Belief über $\theta$ ändert
- Schätze $p(\theta) \approx q_\phi(\theta)$ und nutze $D_{KL}(q_{\phi'}(\theta) \parallel q_\phi(\theta))$ als Explorations-Bonus

---

#### Exploration mit Modellfehlern

**Idee:** $D_{KL}(q_{\phi'}(\theta) \parallel q_\phi(\theta))$ kann auch als Änderung der Netzwerkparameter $\phi$ aufgefasst werden.

**Unabhängig von IG gibt es viele Wege, diese zu messen:**

**1. Stadie et al. 2015:**
- Bilder mit Autoencoder codieren
- Vorhersagemodell auf latentem Raum des Autoencoders
- **Modellfehler als Explorationsbonus nutzen**

**2. Schmidhuber et al.** ("Formal Theory of Creativity, Fun, and Intrinsic Motivation"):
- Explorationsbonus für:
  - Modellfehler
  - Modellgradient
  - Viele andere Varianten

---

### 14. Zusammenfassung: Explorationsklassen in Deep RL

| Klasse | Methoden | Kernidee |
|--------|----------|----------|
| **Optimistische Exploration** | Pseudo-Counts, Hash-Counting, Exemplar Models, Random Network Distillation | Neuer Zustand = guter Zustand, Explorationsbonus |
| **Thompson Sampling** | Bootstrapped DQN | Verteilung über Q-Funktionen, Sampling |
| **Information Gain** | Prediction Gain, VIME, Modellfehler | Maximiere Informationsgewinn |

---

## 15. Offline Reinforcement Learning

### Motivation und Hintergrund

#### Generalization Gap in Deep RL

**Beobachtung:** Deep RL funktioniert auf einer großen Breite von Aufgaben, aber:
- **Generalisierung nicht wie bei Supervised Learning!**

**Ursache:**
- Schwierigkeit von RL: Daten werden **während des Trainings gesammelt**
- **On-policy:** Update der Policy nur mit den neusten Daten
- **Off-policy:** Update der Policy aus einem Replay-Buffer, der aber während des Trainings befüllt und ständig erneuert wird

**Konsequenz:**
- Müssen riesige Datenmengen **pro Trainingslauf** sammeln
- **Idealerweise:** Daten einmal sammeln und wiederverwenden, wie bei Supervised Learning

---

#### Warum funktioniert Deep Learning so gut?

**Formel für Deep Learning Erfolg:**
```
Große Modelle + Große Datensätze = Gute Performance
```

**Frage:** Können wir datenbasierte RL-Algorithmen entwickeln?
- **Auch bekannt als:** "Batch RL"

---

#### Weitere Gründe für Offline RL

**Typisches (Online) RL Setup:**
- Sammeln von Erfahrung durch **Interaktion mit der Umgebung**

**Problem:** In vielen Bereichen **nicht möglich**:
- **Sicherheit:** Keine riskanten Interaktionen erlaubt
- **Kosten:** Echte Roboter teuer, Wear-and-Tear

**Beispiele:**
- Gesundheitswesen (keine riskanten Behandlungen zum Lernen)
- Autonomes Fahren (keine gefährlichen Manöver zum Lernen)

**Offline RL ermöglicht:**
- Aus gesammelten Daten gute Policies zu extrahieren
- **Ohne riskante oder teure Interaktionen**

---

### 16. Formale Definition von Offline RL

#### Datensatz

$$\mathcal{D} = \{(s_t, a_t, s'_t, r_t)\}$$

**Datenverteilung:**
- $s \sim d^{\pi'}(s)$ (unbekannte Verteilung)
- $a \sim \pi^D(a|s)$ (Daten-sammelnde Policy, unbekannt)
- $s' \sim p(s'|s, a)$ (Transition)
- $r = r(s, a)$ (Reward-Funktion)

**Wichtig:** Keine Annahmen an $\pi^D$ (kann beliebig sein)!

#### RL-Zielfunktion

$$\max_\pi \mathbb{E}_{\tau \sim \pi} \left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t)\right]$$

**Aber:** Nur mit Daten aus $\mathcal{D}$!

---

#### Verschiedene Offline RL Aufgabenstellungen

| Aufgabe | Ziel |
|---------|------|
| **Off-Policy Evaluation (OPE)** | Gegeben $\mathcal{D}$, schätze $J(\pi) = \mathbb{E}_\tau[\sum_t r(s_t, a_t)]$ für eine gegebene Policy $\pi$ |
| **Offline Reinforcement Learning** | Gegeben $\mathcal{D}$, lerne die **bestmögliche Policy** $\pi^*$ (bestmöglich gegeben $\mathcal{D}$ – nicht bestmöglich für den MDP!) |

---

### 17. Wie ist Offline RL überhaupt möglich?

#### Intuition: Ordnung aus Chaos

**Frage:** Wie kann man aus einem festen Datensatz eine bessere Policy lernen?

**Antwort 1: Finde die guten Aktionen im Datensatz**
- Datensatz enthält gute und schlechte Aktionen
- **Nicht einfach Imitation Learning!** (BC würde alles kopieren, auch schlechte Aktionen)

**Antwort 2: Generalisierung**
- Gutes Verhalten in einer Situation liefert Beispiele für gutes Verhalten in anderen Situationen

**Antwort 3: "Stitching"**
- **Micro-Stitching:** Teile guten Verhaltens können kombiniert werden
- **Macro-Stitching:** Ganze Trajektorien können neu kombiniert werden

**Beispiel:**
- Trajektorie 1: $A \to B \to C$ (gut)
- Trajektorie 2: $C \to D \to E$ (gut)
- **Offline RL kann lernen:** $A \to B \to C \to D \to E$ (kombiniert beide)
- **Behavioral Cloning kann das nicht!** (würde nur komplette Trajektorien kopieren)

---

#### Offline RL vs. Imitation Learning

**Schlechte Intuition:** "Es ist wie Imitation Learning"

**Bessere Intuition:** "Ordnung aus Chaos"

**Theoretisches Resultat** (Kumar et al. "Should I run Offline RL or BC?"):
- Unter bestimmten Annahmen kann man zeigen, dass **Offline RL selbst mit optimalen Daten besser ist als Imitation Learning**

**Warum?**
- BC kopiert nur
- Offline RL kann **bessere Policies finden** als im Datensatz enthalten (durch Stitching und Generalisierung)

---

### 18. Was macht Offline RL so schwierig?

#### Fundamentales Problem: Counterfactual Queries

**Frage:** "Ist das gut oder schlecht?"

**Problem:** Woher soll der Agent das wissen, wenn er es nicht in den Daten gesehen hat?

**Vergleich Online vs. Offline RL:**

| Online RL | Offline RL |
|-----------|------------|
| Probieren Aktion aus und schauen was passiert | **Keine Möglichkeit** herauszufinden, dass das keine optimale Aktion ist |
| Können Exploration betreiben | Müssen mit **ungesehenen ("out-of-distribution") Aktionen** umgehen |

**Wichtig:** out-of-distribution-actions $\neq$ out-of-sample-actions!
- **OOD-Aktionen:** Aktionen, die von der Daten-Policy $\pi^D$ nicht gewählt werden
- **Out-of-sample:** Andere Samples derselben Verteilung

---

#### Distribution Shift Problem

**Vergleich mit Supervised Learning:**

**Supervised Learning:**
$$\min \mathbb{E}_{x \sim p(x), y \sim p(y|x)} [(f_\theta(x) - y)^2]$$

**Problem:** Adversarial Examples
- Wähle $x^* \leftarrow \arg\max_x f_\theta(x)$
- Gegeben ein $x^*$, ist $f(x^*)$ korrekt?
- $\mathbb{E}_{x \sim p(x), y \sim p(y|x)} [(f_\theta(x) - y)^2]$ ist klein
- **Aber:** $\mathbb{E}_{x \sim \bar{p}(x), y \sim p(y|x)} [(f_\theta(x) - y)^2]$ ist **nicht klein** für beliebige $\bar{p}(x) \neq p(x)$

**Was, wenn $x^* \sim p(x)$?** Nicht unbedingt korrekt!

---

#### Woher kommt der Distribution Shift in RL?

**Q-Learning Update:**
$$Q(s, a) \leftarrow r(s, a) + \max_{a'} Q(s', a')$$

**Offline RL mit Function Approximation:**
$$\min_{\theta} \mathbb{E}_{(s,a) \sim \mathcal{D}} [(Q_\theta(s, a) - y(s, a))^2]$$

**Target:**
$$y(s, a) = r(s, a) + \mathbb{E}_{a' \sim \pi(\cdot|s')} [Q(s', a')]$$

**Problem:**
- Erwarten guten Fit, wenn $\pi_\theta(a|s) = \pi^D(a|s)$ (Behavior-Policy)
- **Aber:** Wir wollen ja, dass $\pi^{RL}$ besser wird als $\pi^D$!
- Sogar: $\pi^{RL} = \arg\max_a \mathbb{E}_{a \sim \pi}[Q(s, a)]$ (**"adversarial attack"**)

**Konsequenz:**
- Policy wird Aktionen wählen, die **nicht im Datensatz** sind
- Q-Funktion wurde auf diesen Aktionen **nicht trainiert**
- **Q-Werte sind unzuverlässig** (Extrapolation!)

---

#### Generalisierungsfehler werden nicht korrigiert

**Online RL:**
1. Beste Aktion laut Q-Function wird gewählt
2. Aktion wird ausgeführt
3. **Wert wird korrigiert, wenn nötig** (durch neues Feedback)

**Offline RL:**
1. Beste Aktion laut Q-Function wird gewählt
2. **Keine Möglichkeit** herauszufinden, dass das keine optimale Aktion ist
3. **Fehler akkumulieren sich!**

**Fazit:** Herausforderungen mit Fehlern des Function Approximators in Standard-RL sind **noch viel schlimmer in Offline RL**!

---

### 19. Offline RL Algorithmen

#### Übersicht

| Kategorie | Methoden | Kernidee |
|-----------|----------|----------|
| **Policy Gradients** | Importance Sampling | Direkte Policy-Optimierung |
| **Policy Constraint** | BRAC, BEAR, BCQ, TD3+BC | Constraint auf Policy-Distanz |
| **Implicit Policy Constraints** | AWR, AWAC, CRR | Advantage-weighted Regression |
| **Conservative Q-Learning** | CQL | Konservative Q-Wert-Schätzung |
| **Implicit Q-Learning** | IQL | Expectile Loss für Value-Funktion |
| **Sequence Modelling** | Decision Transformer, Trajectory Transformer | Transformer-basierte Ansätze |

---

### 20. Offline RL mit Policy Gradients

#### Grundidee

**Policy Gradient Theorem:**
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(s_t, a_t) Q^{\pi_\theta}(s_t, a_t)\right]$$

**Reward-to-go:**
$$\hat{Q}_t = \sum_{t'=t}^T r(s_{t'}, a_{t'})$$

**Monte-Carlo-Schätzung:**
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta(s_{i,t}, a_{i,t}) \hat{Q}(s_{i,t}, a_{i,t})$$

**Problem:** Müssen mit $\pi_\theta$ sampeln - aber wir haben nur Samples von $\pi^D$!

---

#### Importance Sampling

**Lösung:** Importance Sampling zur Korrektur der Verteilung

**Gradient mit Importance Sampling:**
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \left[\prod_{t'=0}^t \frac{\pi_\theta(s_{i,t'}, a_{i,t'})}{\pi^D(s_{i,t'}, a_{i,t'})}\right] \nabla_\theta \log \pi_\theta(s_{i,t}, a_{i,t}) \hat{Q}(s_{i,t}, a_{i,t})$$

**Problem:**
- **Importance Weights exponentiell in $T$!**
- Führt zu **hoher Varianz** in Gradient-Schätzung
- In der Praxis oft nicht verwendbar für lange Episoden

---

### 21. Policy Constraint Methoden

#### Problemstellung

**Problem 1:** Kennen $\pi^D$ nicht, Daten könnten kommen von:
- Demonstrationen von Menschen
- Einer regelbasierten Steuerung
- Vielen früheren RL-Läufen
- Einer Kombination von allem

**Möglichkeiten für Problem 1:**
- Können ein Modell für $\pi^D$ fitten (Behavioral Cloning)
- Methoden, die mit Samples auskommen

**Problem 2:** $D_{KL}$-Constraint ist gleichzeitig **zu pessimistisch und nicht pessimistisch genug**:
- **Nicht pessimistisch genug:** Weil selbst für $D_{KL} < \epsilon$ einzelne Vorhersagen sehr unterschiedlich sein können
- **Zu pessimistisch:** Weil wir eine Policy möchten, die deutlich besser als $\pi^D$ ist
- **Beispiel:** Wenn $\pi^D$ uniforme Zufallspolicy, zwingt $D_{KL}$-Constraint die neue Policy ebenfalls nah an uniform zufällig zu sein

---

#### Explizite Policy Constraints

**Welche Formen sind möglich?**

**1. KL-Divergenz:**
$$D_{KL}(\pi \parallel \pi^D)$$
- **Vorteil:** Leicht zu implementieren
- **Nachteil:** Nicht unbedingt was wir wollen

**2. Support Constraint:**
$$\pi(a|s) \geq 0 \text{ nur, wenn } \pi^D(a|s) \geq \epsilon$$
- **Vorteil:** Viel näher dran an dem, was wir wirklich wollen
- **Nachteil:** Deutlich aufwendiger zu implementieren

**Beispielalgorithmen:**
- **BRAC** (Wu et al., 2019): KL-basierter Constraint
- **BEAR** (Kumar et al., 2019): Support-Constraint
- **BCQ** (Fujimoto et al., 2019): Action-/Support-Constraint
- **TD3+BC** (Fujimoto & Gu, 2021): BC-regularisiertes TD3, minimalistische Baseline mit oft sehr guten Ergebnissen

---

### 22. Implizite Policy Constraints Methoden

#### Lagrange-Multiplikatoren Ansatz

**Optimale Lösung** (mit Lagrange-Multiplikatoren):
$$\pi^*(a|s) = \frac{1}{Z(s)} \pi^D(a|s) \exp\left(\frac{1}{\lambda} A^{\pi^*}(s, a)\right)$$

**Interpretation:**
- Optimale Lösung ist Behavior-Policy $\pi^D$ gewichtet mit $\exp$ von Advantage-Term $A(s, a)$
- **Bessere Aktionen** erhalten in der optimalen Policy **exponentiell mehr Gewicht** gegenüber schlechteren Aktionen

**Problem:** Wir kennen weder $\pi^D(a|s)$ noch $A^{\pi^*}(s, a)$!

---

#### Idee: Näherung durch iteratives Max-Likelihood-Training

**Ansatz:**
$$\pi^{RL}(a|s) = \arg\max_\theta \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[\log \pi_\theta(a|s) \cdot w(s, a)\right]$$

**Gewichte:**
$$w(s, a) = \frac{1}{Z(s)} \pi^D(a|s) \exp\left(\frac{1}{\lambda} A^{\pi_{RL}}(s, a)\right)$$

**Interpretation:** "Gewichtetes Behavioral Cloning"
- Training eines Critics um $A^{\pi_{RL}}$ zu erhalten
- Policy wird auf Daten trainiert, gewichtet nach Advantage

---

#### Loss-Funktionen

**Critic Loss** (für $Q(s, a)$):
$$\mathcal{L}_C(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} [(Q_\theta(s, a) - (r + \gamma V(s')))^2]$$

**Actor Loss** (für $\pi(s, a)$):
$$\mathcal{L}_A(\theta) = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[\log \pi_\theta(a|s) \cdot \exp\left(\frac{1}{\lambda} A(s, a)\right)\right]$$

**Komponenten:**
- $\lambda$: Kontrolliert die Stärke des $D_{KL}$-Constraints (Hyperparameter)
- $Z(s)$: Verschiedene Möglichkeiten, am einfachsten: "weglassen" ($Z(s) = 1$)

**Algorithmen:**
- **AWR** (Advantage Weighted Regression): https://arxiv.org/abs/1910.00177
- **AWAC** (Advantage Weighted Actor Critic): https://arxiv.org/abs/2006.09359
- **CRR** (Critic Regularized Regression): https://arxiv.org/abs/2006.15134

**Hinweis:** Herleitung nimmt explizit stochastische Policy an.

---

### 23. Implicit Q-Learning (IQL)

#### Motivation

**Frage:** Können wir auch im Q-Update OOD-Aktionen vermeiden?

**Standard Q-Learning:**
$$Q(s, a) \leftarrow r(s, a) + \mathbb{E}_{a' \sim \pi(\cdot|s')} [Q(s', a')]$$

**Problem:** $\pi(\cdot|s')$ kann OOD-Aktionen wählen!

---

#### IQL Ansatz

**Idee:** Verwende separates Value-Netzwerk $V(s)$ (weiteres Netzwerk)

**Training von $V(s)$:**
- Erhalte $Q(s', a')$ für verschiedene $a'$ durch Generalisierung über ähnliche $s'$
- Loss zum Training von $V$: $L = \sum_i l(V(s_i), a_i, Q(s_i, a_i))$
- Summe über Daten, d.h. Aktionen $a_i$ kommen von $\pi^D$

**Beispiel MSE-Loss:**
$$l(V(s_i) - Q(s_i, a_i))^2$$
- Gibt uns Erwartungswert von $Q$ unter $\pi^D$ (nicht was wir wollen!)

---

#### Expectile Loss

**Definition:**
$$l_\tau(x) = \begin{cases} (1 - \tau) |x|^2 & (x > 0) \\ \tau |x|^2 & (x \leq 0) \end{cases}$$

**Eigenschaften:**
- **Negative Fehler werden stärker bestraft als positive**
- **Keine Overestimation**, weil wir $Q(s, a)$ nur auf $s, a \in \mathcal{D}$ auswerten

**Bei $l_\tau(x)$ mit $\tau$ groß genug (z.B. $\tau = 0.8$):**
- MSE gibt uns: $\mathbb{E}_{a \sim \pi^D}[Q(s, a)]$ (Durchschnitt)
- **Expectile-Loss liefert:** Wert der besten durch die Daten gestützten Policy

**Vorteil:**
- Verteilung induziert nur durch verschiedene Aktionen
- **Keine OOD-Aktionen nötig!**

---

#### IQL Algorithmus

**"Q-learning with implicit policy improvement"**

**Update-Regeln:**
$$Q(s, a) \leftarrow r(s, a) + V(s')$$
$$V(s) \leftarrow \arg\min_V \sum_{i} l_\tau^{expectile}(V(s_i) - Q(s_i, a_i))$$

**Effektives Verhalten:**
- Bei $l_\tau^{expectile}(x)$ mit $\tau$ groß genug: $V(s)$ approximiert $\max_{a \in \mathcal{D}} Q(s, a)$

**Vorteil:**
- $Q(s, a)$ Updates können nun gemacht werden **ohne out-of-distribution Aktionen zu riskieren**
- Weil nur State-Action-Paare aus dem Datensatz genutzt werden!

---

#### Policy-Extraktion

**Nach Training von $Q(s, a)$:**

**Policy-Extraktion mit advantage-weighted Methode:**
$$\mathcal{L}_\pi(\theta) = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[\log \pi_\theta(a|s) \cdot \exp\left(\frac{1}{\lambda} A(s, a)\right)\right]$$

**Advantage:**
$$A(s, a) = Q(s, a) - V(s)$$

**Wichtig:**
- **Einfach greedy($Q$) funktioniert nicht!**
- Dadurch potentiell Aktionswerte außerhalb der Datenverteilung abgefragt

**Quelle:** Kostrikov, Nair, Levine. "Offline Reinforcement Learning with Implicit Q-Learning" (2021)

---

### 24. Conservative Q-Learning (CQL)

#### Grundidee

**Alternativer Ansatz für das OOD-Problem:**

**CQL Objective:**
$$Q^*_{CQL} = \arg\min_Q \left\{\max_\mu \alpha \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(\cdot|s)} [Q(s, a)] - \alpha \mathbb{E}_{(s,a) \sim \mathcal{D}} [Q(s, a)] + \mathcal{L}_{TD}(Q)\right\}$$

**Komponenten:**
1. $\alpha \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(\cdot|s)} [Q(s, a)]$: **Maximiert Q-Werte** für alle Aktionen (führt zu Overestimation)
2. $-\alpha \mathbb{E}_{(s,a) \sim \mathcal{D}} [Q(s, a)]$: **Reduziert Q-Werte** für Daten aus dem Datensatz
3. $\mathcal{L}_{TD}(Q)$: Regulärer Q-Learning Loss

**Effekt:**
- **Reduziert zu große Q-Werte** für OOD-Aktionen
- Es lässt sich zeigen: $Q_{CQL} \leq Q_{true}$ wenn $\alpha$ groß genug

---

#### Verbesserte Schranke

**Problem:** Vorige Schranke tendenziell zu pessimistisch

**Bessere Schranke:**
$$Q^*_{CQL} = \arg\min_Q \left\{\max_\mu \alpha \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(\cdot|s)} [Q(s, a)] - \alpha \mathbb{E}_{(s,a) \sim \mathcal{D}} [Q(s, a)] + \mathcal{L}_{TD}(Q)\right\}$$

**Interpretation:**
- **Reduziert Q-Werte $\forall s, a$** (erster Term)
- **Erhöht Q-Werte, wenn $(s, a) \in \mathcal{D}$** (zweiter Term hebt Reduktion auf)

**Resultat:**
- **Effektiv nur Q-Werte für Aktionen außerhalb des Datensatzes reduziert**

**Theoretische Garantie:**
- Es gilt **nicht mehr:** $Q_{CQL}(s, a) \leq Q_{true}(s, a)$ für alle $(s, a)$
- **Stattdessen gilt:** $\mathbb{E}_{a \sim \pi}[Q_{CQL}(s, a)] \leq \mathbb{E}_{a \sim \pi}[Q_{true}(s, a)]$ für alle $s \in \mathcal{D}$

**Quelle:** Kumar, Zhou, Tucker, Levine. "Conservative Q-Learning for Offline Reinforcement Learning" (2020)

---

#### CQL Algorithmus

**CQL ist ein Actor-Critic Algorithmus:**

**1. Update $Q_{CQL}$ durch Minimierung von $\mathcal{L}_{CQL}(Q)$ und Daten aus $\mathcal{D}$**

**2. Update Policy $\pi$:**

**Bei deterministischer Policy:**
$$\pi(a|s) = \arg\max_a Q(s, a)$$

**Bei stochastischer Policy:**
$$\theta \leftarrow \theta + \alpha \nabla_\theta \sum_{s \in \mathcal{D}} \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} [Q_{CQL}(s, a)]$$
(Policy Gradient Update)

---

#### Regularizer für $\mu$

**Problem:** Ohne Regularizer:
$$\max_\mu Q(s, a) \rightarrow \max_a Q(s, a)$$
- Nur eine Aktion $a$ wird betrachtet
- $Q$ wird nur an dieser Stelle "gefixt"

**Typischerweise (insbesondere bei kontinuierlichen Aktionsräumen):**
- $Q(s, a)$ ist für viele Aktionen außerhalb von $\mathcal{D}$ zu groß
- **Wir wollen $Q(s, a)$ überall dort verringern**
- **Brauchen eine breite Verteilung über alle Aktionen $a$, für die $Q(s, a)$ groß ist**

**Typische Wahl:**
$$\mathcal{R}(\mu) = \mathbb{E}_{s \sim \mathcal{D}} [\mathcal{H}(\mu(\cdot|s))]$$
(Maximum Entropy Regularization)

**Resultiert in:**
$$\mu(a|s) \propto \exp(Q(s, a)/\tau)$$

**Berechnung:**
- **Für diskrete Aktionen:** Leicht zu berechnen
  $$\mathbb{E}_{a \sim \mu(\cdot|s)} [Q(s, a)] = \tau \log \sum_a \exp\left(\frac{Q(s, a)}{\tau}\right)$$
- **Für kontinuierliche Aktionen:** Importance Sampling aus beliebiger Policy

---

### 25. Offline RL als Sequence Modelling

#### Grundidee

**Offline-Daten:** $(s_1, a_1, r_1, s_2, a_2, r_2, \dots)$

**Ansatz:** Sequenzmodellierung (wie bei Sprachmodellen):
- Lerne $p(\text{next token} | \text{history})$
- **Kein Bootstrapping** → keine OOD-Q-Fehler!

---

#### Decision Transformer (DT)

**Paper:** https://arxiv.org/abs/2106.01345

**Input-Sequenz:**
$$(R_1, s_1, a_1, R_2, s_2, a_2, \dots)$$

**Return-to-go:**
$$R_t = \sum_{t'=t}^T r_{t'}$$
(Ab Zeit $t$ kumulierter Reward)

**Loss:**
$$\mathcal{L} = \|a_t - \hat{a}_t\|^2$$
- $a_t$: Dataset-Action
- $\hat{a}_t$: Modellvorhersage

**Inference:**
1. Setze gewünschten Return $R_1 = R^*$ (Ziel-Return)
2. **Modell-Input:** $(R^*, s_1)$
3. **Modell-Output:** $a_1$
4. Environment → $(s_2, r_1)$
5. Update gewünschter Return: $R_2 = R^* - r_1$
6. **Nächster Modell-Input:** $(R^*, s_1, a_1, R_2, s_2)$
7. **Modell-Output:** $a_2$
8. Wiederhole...

**Interpretation:** "Return-konditioniertes Behavioral Cloning"

---

#### Trajectory Transformer (TT)

**Paper:** https://arxiv.org/abs/2106.02039

**Input-Sequenz:**
$$(s_1, a_1, r_1, s_2, a_2, r_2, \dots)$$

**Transformer lernt:**
$$p(\text{next token} | \text{history})$$

**Ansatz:**
- **Generatives Weltmodell**
- Policy wird nicht direkt gelernt → müssen Aktionen sampeln
- **Planning:** Die Folgen vorhersagen
- **Beste Aktionsfolgen mit Beam Search finden**

---

#### Vergleich DT vs. TT

| Aspekt | Decision Transformer | Trajectory Transformer |
|--------|---------------------|------------------------|
| **Vorteil** | Einfacher, schneller, kein Beam Search zur Inference-Zeit | Trajectory Stitching funktioniert (DT kann das meist nicht) |
| **Inference** | Direkt Aktionen generieren | Beam Search nötig |
| **Stitching** | Meist nicht möglich | Funktioniert gut |
| **Komplexität** | Niedriger | Höher |

---

### 26. Frameworks & Daten

#### Offline RL Bibliotheken

| Bibliothek | Beschreibung |
|------------|-------------|
| **d3rlpy** | Umfassende Offline-RL-Bibliothek für Python |
| **CORL** | Continuous Offline RL Library |
| **TorchRL** | Teilweise, Lossfunktionen für CQL, IQL, TD3+BC |

#### Offline RL Datensätze

| Datensatz | Beschreibung |
|-----------|-------------|
| **D4RL** | Klassischer Offline-RL Benchmark (MuJoCo, Maze2D, etc.) |
| **Minari** | Gepflegt von Farama Foundation (gymnasium u.v.m.), moderner Dataset-Standard für RL, enthält viele portierte D4RL-Datensätze |
| **RL Unplugged** | Verschiedene Domänen |
| **RoboNet, Bridge** | Große Robotik-Datensätze mit Daten von echten Robotern |

#### Typische Domänen

- **Locomotion, Manipulation** (Hauptanwendung)
- **Atari** ("klassisches Online RL", auch für Offline verfügbar)

#### Typischer Workflow

1. Datensatz laden
2. Algorithmus wählen (CQL, IQL, TD3+BC, etc.)
3. Offline trainieren
4. Auf Benchmark-Umgebung evaluieren

---

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ UCB (Upper Confidence Bound) - SEHR WICHTIG

**Warum:** Optimale Exploration für Bandits, theoretisch fundiert

**Formel:**
$$a = \arg\max_a \left[\hat{\mu}_a + \sqrt{\frac{2 \ln(T)}{N(a)}}\right]$$

**Intuition:** Optimismus bei Unsicherheit

**Klausurfrage:** "Wie funktioniert UCB und warum ist es optimal für Bandits?"

---

### ✅ Thompson Sampling - SEHR WICHTIG

**Warum:** Elegante Bayesianische Methode, empirisch sehr stark

**Algorithmus:**
1. Sample $\theta \sim \hat{p}(\theta)$
2. $a = \arg\max_a \mathbb{E}_\theta[r|a]$
3. Update Posterior

**Klausurfrage:** "Erklären Sie Thompson Sampling und den Unterschied zu UCB!"

---

### ✅ Pseudo-Counts - WICHTIG

**Warum:** Skaliert auf große Zustandsräume

**Formel:**
$$\hat{N}(s) = \frac{p_\theta(s)}{p_{\theta'}(s) - p_\theta(s)}$$

**Intuition:** Density Model als Näherung von Besuchszahlen

**Klausurfrage:** "Was sind Pseudo-Counts und wie berechnet man sie?"

---

### ✅ Offline RL Problem - SEHR WICHTIG

**Warum:** Wichtige praktische Anwendung, fundamentales Problem

**Kernproblem:** Distribution Shift, Counterfactual Queries

**Frage:** "Was ist das Hauptproblem bei Offline RL und warum ist es schwieriger als Online RL?"

**Antwort:**
- **Distribution Shift:** Policy wählt OOD-Aktionen
- **Kein Online-Feedback:** Fehler können nicht korrigiert werden
- **Extrapolation:** Q-Funktion muss auf ungesehenen Aktionen extrapolieren

---

### ✅ Conservative Q-Learning (CQL) - WICHTIG

**Warum:** Wichtiger Algorithmus für Offline RL

**Objective:**
$$\mathcal{L}_{CQL}(Q) = \alpha \mathbb{E}_{s,a \sim \mu}[Q(s,a)] - \alpha \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)] + \mathcal{L}_{TD}(Q)$$

**Intuition:** Reduziere Q-Werte für OOD-Aktionen

**Klausurfrage:** "Wie funktioniert CQL und welches Problem löst es?"

---

### ✅ Implicit Q-Learning (IQL) - WICHTIG

**Warum:** Vermeidet OOD-Aktionen durch Expectile Loss

**Expectile Loss:**
$$l_\tau(x) = \begin{cases} (1-\tau)|x|^2 & (x > 0) \\ \tau|x|^2 & (x \leq 0) \end{cases}$$

**Klausurfrage:** "Wie vermeidet IQL das OOD-Problem?"

---

### ✅ Decision Transformer - GRUNDWISSEN

**Warum:** Moderner Ansatz, Verbindung zu Transformern

**Idee:** Return-konditioniertes Behavioral Cloning

**Klausurfrage:** "Was ist der Decision Transformer Ansatz?"

---

## Formeln/Algorithmen (wichtig)

### UCB
$$a = \arg\max_a \left[\hat{\mu}_a + \sqrt{\frac{2 \ln(T)}{N(a)}}\right]$$

### Thompson Sampling
$$\theta \sim \hat{p}(\theta)$$
$$a = \arg\max_a \mathbb{E}_\theta[r|a]$$

### Pseudo-Count
$$\hat{N}(s) = \frac{p_\theta(s)}{p_{\theta'}(s) - p_\theta(s)}$$

### Explorations-Bonus
$$r^+(s, a) = r(s, a) + B(N(s))$$
$$B(N) = \sqrt{\frac{2 \ln n}{N(s)}} \text{ (UCB)}$$

### Regret
$$\text{Reg}(T) = T \cdot \mathbb{E}[r(a^*)] - \sum_{t=1}^T \mathbb{E}[r(a_t)]$$

### Information Gain
$$IG(z, y|a) = \mathbb{E}_y[H(\hat{p}(z)) - H(\hat{p}(z|y))|a]$$

### Expectile Loss (IQL)
$$l_\tau(x) = \begin{cases} (1-\tau)|x|^2 & (x > 0) \\ \tau|x|^2 & (x \leq 0) \end{cases}$$

### CQL Objective
$$\mathcal{L}_{CQL}(Q) = \alpha \mathbb{E}_{s,a \sim \mu}[Q(s,a)] - \alpha \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)] + \mathcal{L}_{TD}(Q)$$

### Advantage-Weighted Policy (AWR/AWAC)
$$\pi^*(a|s) = \frac{1}{Z(s)} \pi^D(a|s) \exp\left(\frac{1}{\lambda} A(s, a)\right)$$

### Decision Transformer Input
$$(R_1, s_1, a_1, R_2, s_2, a_2, \dots)$$
$$R_t = \sum_{t'=t}^T r_{t'}$$

---

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte

1. **Exploration ist schwierig** weil Agenten Regeln nicht kennen und nur durch Ausprobieren lernen
2. **Bandits** sind einfachste Explorationsprobleme, theoretisch gut verstanden
3. **UCB:** Optimistisch bei Unsicherheit, optimal für Bandits ($O(\log T)$ Regret)
4. **Thompson Sampling:** Bayesianisch, sample aus Posterior, empirisch sehr stark
5. **Information Gain:** Maximiere Lerngewinn, Balance zwischen Suboptimalität und Information
6. **Pseudo-Counts:** Zählen in hochdimensionalen Räumen durch Density Models
7. **Bootstrapped DQN:** Kohärente Exploration über ganze Episode
8. **Offline RL:** Lernen ohne Interaktion, Distribution Shift Problem
9. **CQL:** Konservative Q-Werte für OOD-Aktionen
10. **IQL:** Expectile Loss vermeidet OOD-Aktionen
11. **Decision Transformer:** Sequence Modelling Ansatz, kein Bootstrapping

### ⚠️ Häufige Fehler

1. **UCB-Formel verwechseln** (Nenner vs. Zähler)
2. **Thompson Sampling vs. UCB** nicht unterscheiden können
3. **Pseudo-Count Berechnung** nicht verstehen
4. **Nicht verstehen, warum Offline RL schwierig ist** (Distribution Shift!)
5. **Denken Offline RL = Imitation Learning** (falsch! Offline RL kann besser sein)
6. **CQL Objective falsch interpretieren** (welcher Term macht was?)
7. **Expectile Loss Intuition** nicht verstehen (warum asymmetrisch?)

### 📝 Prüfungsrelevante Fragen

1. **Was ist der Unterschied zwischen Exploration und Exploitation?**
   - Exploration: Neue Dinge ausprobieren (langfristiger Gewinn)
   - Exploitation: Bekannte gute Aktionen nutzen (kurzfristiger Gewinn)

2. **Wie funktioniert UCB und warum ist es optimal?**
   - $a = \arg\max_a [\hat{\mu}_a + \sqrt{2 \ln(T) / N(a)}]$
   - Optimismus bei Unsicherheit
   - $\text{Reg}(T) = O(\log T)$ beweisbar optimal

3. **Was ist Thompson Sampling?**
   - Sample $\theta \sim \hat{p}(\theta)$
   - Wähle optimale Aktion unter Sample
   - Update Posterior
   - Probability matching Ansatz

4. **Was sind Pseudo-Counts und warum braucht man sie?**
   - Näherung von Besuchszahlen in großen Zustandsräumen
   - $\hat{N}(s) = p_\theta(s) / (p_{\theta'}(s) - p_\theta(s))$
   - Density Model als Count-Ersatz

5. **Was ist das Hauptproblem bei Offline RL?**
   - Distribution Shift: Policy wählt OOD-Aktionen
   - Keine Online-Interaktion zur Korrektur
   - Q-Funktion extrapolieren auf ungesehenen Aktionen

6. **Wie funktioniert CQL?**
   - Minimiert Q-Werte für OOD-Aktionen
   - $\mathcal{L}_{CQL} = \alpha \mathbb{E}_\mu[Q] - \alpha \mathbb{E}_\mathcal{D}[Q] + \mathcal{L}_{TD}$
   - Konservative Q-Wert-Schätzung

7. **Wie vermeidet IQL das OOD-Problem?**
   - Expectile Loss für Value-Funktion
   - Asymmetrische Bestrafung (positive Fehler stärker bestraft)
   - $V(s)$ approximiert beste in-Daten Policy

8. **Was ist der Unterschied zwischen Decision Transformer und Trajectory Transformer?**
   - DT: Return-konditioniertes BC, einfacher, kein Beam Search
   - TT: Generatives Weltmodell, kann Stitching, Beam Search nötig

9. **Warum ist Offline RL besser als Behavioral Cloning?**
   - BC kopiert nur
   - Offline RL kann durch Stitching bessere Policies finden
   - Theoretisch gezeigt unter bestimmten Annahmen

10. **Was ist Information-Directed Sampling?**
    - Balance zwischen Suboptimalität $\Delta(a)$ und Information Gain $g(a)$
    - $a = \arg\min_a \Delta(a)^2 / g(a)$
    - Wähle keine suboptimalen Aktionen, wähle keine Aktionen ohne Lerngewinn

---

## Tipps für die Klausur

### ✅ Prioritäten:
- **UCB und Thompson Sampling** verstehen (wichtigste!)
- **Offline RL Problem** erklären können (Distribution Shift)
- **CQL und IQL** Grundideen kennen
- **Pseudo-Counts** Konzept verstehen

### ✅ Typische Frage-Typen:
- "Erklären Sie..." - UCB, Thompson Sampling, Offline RL Problem
- "Was ist der Unterschied..." - Online vs. Offline RL, CQL vs. IQL
- "Wie funktioniert..." - Pseudo-Counts, Decision Transformer
- "Warum..." - Offline RL schwierig, CQL konservativ

### ✅ Wichtige Konzepte (auswendig):
- **UCB-Formel:** $\hat{\mu}_a + \sqrt{2 \ln(T) / N(a)}$
- **Thompson Sampling:** Sample aus Posterior, handle optimal
- **Regret:** Differenz zur optimalen Policy
- **Distribution Shift:** $p_{train} \neq p_{test}$
- **CQL:** Konservative Q-Werte für OOD

---

## Querverweise

| Thema | Verbindung zu |
|-------|--------------|
| **UCB/Thompson** | RL-Teil-1: $\epsilon$-greedy Exploration |
| **Pseudo-Counts** | XAI: Dichtemodellierung |
| **Offline RL** | Imitation Learning: DAgger, Behavioral Cloning |
| **CQL/IQL** | RL-Teil-1: Q-Learning, DQN |
| **Decision Transformer** | Transformers: Self-Attention, Sequence Modelling |

---

**Erstellt:** 17.03.2026  
**Aktualisiert:** 17.03.2026 (sehr ausführlich erweitert basierend auf PDF + Beispielfragen)  
**Basierend auf:** AdvancedML-08-RL-Teil-2.pdf (~76 Seiten) + Cheatsheet + Beispielfragen  
**Klausurrelevanz:** 🔴 SEHR HOCH - Exploration und Offline RL sind wichtige Themen
