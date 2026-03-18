# Selbsttest Tag 6: Exploration, Imitation Learning & RLHF

**Umfang:** Ausführlicher Test zu Exploration in Deep RL, Offline RL, Imitation Learning und Alignment (RLHF/DPO)  
**Zeitansatz:** 50-70 Minuten  
**Hinweis:** Antworten nicht einfach ablesen - erst selbst überlegen, dann nachschlagen!

---

## Teil A: Grundkonzepte & Verständnis (12 Fragen)

### 1. Exploration vs Exploitation

**Frage 1:**  
Erklären Sie das fundamentale Dilemma von Exploration vs Exploitation. Warum kann ein Agent nicht einfach immer die beste bekannte Aktion wählen?

**Frage 2:**  
Was versteht man unter "Regret" im Kontext von Exploration? Geben Sie die mathematische Definition an und erklären Sie die einzelnen Terme.

**Frage 3:**  
Warum ist das Multi-armed Bandit Problem theoretisch lösbar, während große, unendliche MDPs mit kontinuierlichen Räumen theoretisch unlösbar sind?

---

### 2. Bandit-Algorithmen

**Frage 4:**  
Erklären Sie die Grundidee von UCB (Upper Confidence Bound). Was bedeutet "Optimismus bei Unsicherheit" konkret?

**Frage 5:**  
Schreiben Sie die UCB-Formel aus dem Gedächtnis auf. Was passiert mit dem Explorations-Bonus, wenn eine Aktion häufiger gewählt wird?

**Frage 6:**  
Wie funktioniert Thompson Sampling? Beschreiben Sie den Algorithmus in 3 Schritten und erklären Sie den Unterschied zu UCB.

**Frage 7:**  
Was ist Information Gain und wie wird er berechnet? Erklären Sie die Formel $IG(z, y|a) = \mathbb{E}_y[H(\hat{p}(z)) - H(\hat{p}(z|y))|a]$.

---

### 3. Exploration in Deep RL

**Frage 8:**  
Was sind Pseudo-Counts und warum werden sie benötigt? Wie berechnet man $\hat{N}(s)$ aus $p_\theta(s)$ und $p_{\theta'}(s)$?

**Frage 9:**  
Erklären Sie den Unterschied zwischen $\epsilon$-greedy Exploration und Bootstrapped DQN. Warum führt Bootstrapped DQN zu "kohärenter Exploration"?

**Frage 10:**  
Was ist Random Network Distillation (RND)? Wie wird der Explorations-Bonus berechnet und war funktioniert die Methode?

---

### 4. Imitation Learning Grundlagen

**Frage 11:**  
Was ist das Distributional Shift Problem beim Behavioral Cloning? Warum führt es zu quadratischer Fehlerakkumulation $O(\epsilon T^2)$ statt linearer $O(\epsilon T)$?

**Frage 12:**  
Beschreiben Sie den DAgger (Dataset Aggregation) Algorithmus in 5 Schritten. Welche theoretische Garantie bietet DAgger?

---

## Teil B: Formeln & Berechnungen (10 Fragen)

### 5. UCB und Thompson Sampling

**Frage 13:**  
Berechnen Sie den UCB-Wert für eine Aktion mit:
- Geschätzter durchschnittlicher Reward $\hat{\mu}_a = 0.7$
- Anzahl Besuche $N(a) = 10$
- Gesamtzahl Schritte $T = 1000$

**Frage 14:**  
In einem 3-Armed Bandit Problem mit Bernoulli-Verteilung haben die Arme folgende Posterior-Parameter (Beta-Verteilung):
- Arm 1: $\alpha = 5, \beta = 2$
- Arm 2: $\alpha = 3, \beta = 3$
- Arm 3: $\alpha = 2, \beta = 5$

Bei einem Thompson Sampling Sample erhalten Sie: $\theta_1 = 0.8, \theta_2 = 0.5, \theta_3 = 0.3$. Welche Aktion wird gewählt?

**Frage 15:**  
Berechnen Sie den Pseudo-Count für einen Zustand mit:
- $p_\theta(s) = 0.01$
- $p_{\theta'}(s) = 0.0105$

Was bedeutet das Ergebnis praktisch?

---

### 6. Offline RL Formeln

**Frage 16:**  
Schreiben Sie das CQL (Conservative Q-Learning) Objective aus dem Gedächtnis auf. Was bewirkt der erste Term? Was bewirkt der zweite Term?

**Frage 17:**  
Was ist der Expectile Loss? Schreiben Sie die Formel für $l_\tau(x)$ auf und erklären Sie, warum er asymmetrisch ist.

**Frage 18:**  
Gegeben die Advantage-Weighted Policy Formel:
$$\pi^*(a|s) = \frac{1}{Z(s)} \pi^D(a|s) \exp\left(\frac{1}{\lambda} A(s, a)\right)$$

Was passiert mit Aktionen, die einen hohen Advantage haben? Was passiert, wenn $\lambda$ sehr groß wird?

**Frage 19:**  
Berechnen Sie den Information-Directed Sampling Wert für eine Aktion mit:
- Erwartete Suboptimalität $\Delta(a) = 0.5$
- Information Gain $g(a) = 0.25$

Vergleichen Sie mit einer zweiten Aktion: $\Delta(a) = 0.3, g(a) = 0.1$. Welche Aktion wird gewählt?

**Frage 20:**  
Schreiben Sie die DPO (Direct Preference Optimization) Loss-Funktion auf. Was ist der Unterschied zum RLHF-Ansatz mit separatem Reward-Modell?

---

## Teil C: Vergleiche & Analyse (10 Fragen)

### 7. Algorithmen-Vergleiche

**Frage 21:**  
Vergleichen Sie UCB, Thompson Sampling und Information-Directed Sampling in einer Tabelle:

| Aspekt | UCB | Thompson Sampling | Information-Directed Sampling |
|--------|-----|-------------------|-------------------------------|
| Grundidee | ? | ? | ? |
| Bayesianisch? | ? | ? | ? |
| Explorations-Bonus | ? | ? | ? |
| Regret-Bound | ? | ? | ? |

**Frage 22:**  
Was ist der Unterschied zwischen Online RL und Offline RL? Nennen Sie jeweils 2 Vorteile und 2 Nachteile.

**Frage 23:**  
Vergleichen Sie Behavioral Cloning und DAgger:
- Welche Daten werden für das Training verwendet?
- Was ist das theoretische Regret-Verhalten?
- Welche praktischen Nachteile hat DAgger?

---

### 8. Offline RL Methoden

**Frage 24:**  
Vergleichen Sie CQL und IQL:
- Welches Problem lösen beide?
- Wie vermeidet CQL OOD-Aktionen?
- Wie vermeidet IQL OOD-Aktionen?
- Welche Methode ist einfacher zu implementieren?

**Frage 25:**  
Erklären Sie den Unterschied zwischen:
- Policy Constraint Methoden (BEAR, BCQ)
- Implicit Policy Constraints (AWR, AWAC)
- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL)

**Frage 26:**  
Warum funktioniert Decision Transformer (DT) kein Trajectory Stitching, während Trajectory Transformer (TT) das kann? Was ist der praktische Unterschied?

---

### 9. Alignment und Reasoning

**Frage 27:**  
Vergleichen Sie RLHF und DPO:
- Welche Komponenten benötigt RLHF?
- Welche Komponenten benötigt DPO?
- Was ist der Hauptvorteil von DPO?
- Gibt es Nachteile von DPO gegenüber RLHF?

**Frage 28:**  
Was ist der Unterschied zwischen:
- Chain-of-Thought Prompting
- Tree of Thoughts
- Self-Consistency

Wann sollte man welche Methode verwenden?

**Frage 29:**  
Erklären Sie RAG (Retrieval-Augmented Generation). Wie unterscheidet es sich von einem Standard-LLM und welche Vorteile bietet es?

**Frage 30:**  
Was ist Constitutional AI und wie unterscheidet es sich von RLHF? Nennen Sie die beiden Hauptphasen (SL-CAI und RL-CAI).

---

## Teil D: Praktische Anwendungen & Edge Cases (8 Fragen)

### 10. Praktische Implementierung

**Frage 31:**  
Sie trainieren einen Agenten für ein komplexes Spiel wie Montezuma's Revenge. Warum ist Exploration hier besonders schwierig? Welche Exploration-Methode würden Sie empfehlen und warum?

**Frage 32:**  
Sie haben einen Datensatz von menschlichen Demonstrationen für ein autonomes Fahrsystem. Warum könnte reines Behavioral Cloning scheitern? Wie könnten Sie das Problem lösen (3 Methoden nennen)?

**Frage 33:**  
In Offline RL: Warum ist das Distribution Shift Problem besonders gravierend? Warum können Fehler nicht korrigiert werden wie im Online RL?

**Frage 34:**  
Was ist "Reward Hacking" im Kontext von RLHF? Wie kann man es verhindern?

---

### 11. Tiefes Verständnis & Edge Cases

**Frage 35:**  
Warum funktioniert die UCB-Formel $\sqrt{\frac{2 \ln(T)}{N(a)}}$? Erklären Sie intuitiv, warum der Bonus mit $N(a)$ abnimmt und mit $\ln(T)$ zunimmt.

**Frage 36:**  
Was passiert bei Thompson Sampling, wenn der Posterior sehr eng wird (hohe Konfidenz)? Wie unterscheidet sich das Verhalten von UCB in diesem Fall?

**Frage 37:**  
Warum funktioniert IQL mit Expectile Loss, aber nicht mit einfachem MSE Loss? Was würde bei MSE passieren?

**Frage 38:**  
Erklären Sie, warum CQL konservativ ist (untere Schranke für Q-Werte), aber nicht zu pessimistisch. Was garantiert die verbesserte Schranke?

**Frage 39:**  
Was ist das "Exposure Bias" Problem beim Training von Seq2Seq-Modellen und wie hängt es mit dem Distributional Shift Problem zusammen?

**Frage 40:**  
Warum kann Offline RL theoretisch besser sein als Behavioral Cloning, selbst wenn die Daten von einer optimalen Policy stammen? Erklären Sie das Konzept "Trajectory Stitching".

---

## Antworten & Lösungen

<details>
<summary>Klicken Sie hier, um die Antworten anzuzeigen</summary>

### Teil A Antworten

**A1:** Exploration bedeutet, neue Aktionen auszuprobieren, um potenziell höhere Rewards zu entdecken (langfristiger Gewinn). Exploitation bedeutet, die aktuell beste bekannte Aktion zu wählen (kurzfristiger Gewinn). Ein Agent, der immer exploitiert, könnte in einem lokalen Optimum stecken bleiben und nie die globale optimale Policy finden.

**A2:** Regret misst die Differenz zur optimalen Policy:
$$\text{Reg}(T) = T \cdot \mathbb{E}[r(a^*)] - \sum_{t=1}^T \mathbb{E}[r(a_t)]$$
- $T \cdot \mathbb{E}[r(a^*)]$: Erwarteter Return der optimalen Policy
- $\sum_{t=1}^T \mathbb{E}[r(a_t)]$: Tatsächlich erzielter Return
- Ziel: Minimiere Regret

**A3:** Bandits sind 1-step stateless Probleme - es gibt keine Zustandsübergänge zu modellieren. MDPs haben Zustände, Transitionen und zeitlich ausgedehnte Sequenzen. In großen/kontinuierlichen Räumen ist die Zustandsmenge zu groß für exakte Lösungen.

**A4:** "Optimismus bei Unsicherheit" bedeutet, dass wir unsere Reward-Schätzung um einen Bonus erhöhen, der mit der Unsicherheit skaliert. Unbekannte Aktionen (wenige Besuche) bekommen einen hohen Bonus und werden bevorzugt ausprobiert, bis ihre Qualität sicher bekannt ist.

**A5:** UCB-Formel:
$$a = \arg\max_a \left[\hat{\mu}_a + \sqrt{\frac{2 \ln(T)}{N(a)}}\right]$$
Wenn $N(a)$ zunimmt, nimmt der Bonus $\sqrt{\frac{2 \ln(T)}{N(a)}}$ ab (weniger Unsicherheit bei mehr Besuchen).

**A6:** Thompson Sampling Algorithmus:
1. Sample $\theta_1, \dots, \theta_K \sim \hat{p}(\theta_1, \dots, \theta_K)$ aus dem Posterior
2. Wähle optimale Aktion unter dem Sample: $a = \arg\max_a \mathbb{E}_\theta[r|a]$
3. Update Posterior basierend auf Beobachtung

Unterschied zu UCB: Thompson Sampling ist bayesianisch (sampling aus Verteilung), UCB ist frequentistisch (expliziter Bonus).

**A7:** Information Gain misst, wie viel wir über eine latente Variable $z$ lernen, wenn wir Aktion $a$ ausführen und Beobachtung $y$ machen. $H(\hat{p}(z))$ ist die Entropie vor der Beobachtung, $H(\hat{p}(z|y))$ ist die Entropie danach. Der Erwartungswert über alle möglichen $y$ gibt den erwarteten Informationsgewinn.

**A8:** Pseudo-Counts sind eine Näherung von Besuchszählern in großen Zustandsräumen, wo echtes Zählen unmöglich ist. Berechnung:
$$\hat{N}(s) = \frac{p_\theta(s)}{p_{\theta'}(s) - p_\theta(s)}$$
Wobei $p_\theta$ das alte und $p_{\theta'}$ das neue Density Model ist. Das Ergebnis schätzt, wie oft der Zustand "effektiv" besucht wurde.

**A9:** $\epsilon$-greedy wählt zufällige Aktionen unabhängig vom Zustand, was zu inkohärentem Verhalten führt (Oszillieren). Bootstrapped DQN trainiert mehrere Q-Netzwerke auf Bootstrap-Samples des Datensatzes und wählt zufällig eines für eine ganze Episode. Dies führt zu kohärenter Exploration - der Agent folgt einer konsistenten (wenn auch zufälligen) Policy für die gesamte Episode.

**A10:** RND verwendet ein zufälliges Netzwerk $f^*$ (fixe zufällige Gewichte) und trainiert ein Netzwerk $f_\phi$, dieses zu approximieren. Der Bonus ist $\|f_\phi(s) - f^*(s)\|^2$. Hoher Fehler bedeutet, dass der Zustand neu ist (Netzwerk wurde noch nicht auf diesem Zustand trainiert).

**A11:** Beim Behavioral Cloning wird die Policy auf Daten der Experten-Policy $\pi^*$ trainiert, aber bei der Ausführung wird die gelernte Policy $\pi_\theta$ verwendet. Dies führt zu einem Distributional Shift: $p_{train} \neq p_{test}$. Kleine Fehler bringen den Agenten in Zustände, die nicht in den Trainingsdaten enthalten sind, wo die Policy weitere Fehler macht. Die Fehler akkumulieren sich über die Zeit quadratisch: $O(\epsilon T^2)$ statt $O(\epsilon T)$.

**A12:** DAgger Algorithmus:
1. Trainiere $\pi_\theta$ auf Experten-Daten $\mathcal{D}$
2. Führe $\pi_\theta$ aus, erhalte Datensatz $\mathcal{D}_\pi$
3. Experte labelt $\mathcal{D}_\pi$ mit korrekten Aktionen
4. Aggregiere: $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_\pi$
5. Wiederhole ab Schritt 1

Garantie: Lineare Fehlerakkumulation $O(T)$ statt $O(T^2)$.

### Teil B Antworten

**A13:** UCB-Wert:
$$0.7 + \sqrt{\frac{2 \ln(1000)}{10}} = 0.7 + \sqrt{\frac{2 \cdot 6.907}{10}} = 0.7 + \sqrt{1.381} = 0.7 + 1.175 = 1.875$$

**A14:** Bei Thompson Sampling wählt man die Aktion mit dem höchsten gesampelten $\theta$. Hier: Arm 1 mit $\theta_1 = 0.8$ (höchster Wert).

**A15:** Pseudo-Count:
$$\hat{N}(s) = \frac{0.01}{0.0105 - 0.01} = \frac{0.01}{0.0005} = 20$$
Der Zustand wurde effektiv etwa 20 Mal besucht. Praktisch bedeutet dies, dass wir einen moderaten Explorations-Bonus vergeben (nicht neu, aber auch nicht häufig).

**A16:** CQL Objective:
$$\mathcal{L}_{CQL}(Q) = \alpha \mathbb{E}_{s,a \sim \mu}[Q(s,a)] - \alpha \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)] + \mathcal{L}_{TD}(Q)$$
- Erster Term: Maximiert Q-Werte für alle Aktionen (führt zu Overestimation)
- Zweiter Term: Reduziert Q-Werte für Daten aus dem Datensatz (hebt Overestimation auf)
- Effekt: Nur Q-Werte für OOD-Aktionen werden reduziert

**A17:** Expectile Loss:
$$l_\tau(x) = \begin{cases} (1-\tau)|x|^2 & (x > 0) \\ \tau|x|^2 & (x \leq 0) \end{cases}$$
Asymmetrisch, weil positive und negative Fehler unterschiedlich gewichtet werden. Bei $\tau > 0.5$ werden negative Fehler (Unterschreitung) stärker bestraft als positive Fehler (Überschätzung).

**A18:** Aktionen mit hohem Advantage $A(s,a)$ bekommen exponentiell mehr Gewicht in der Policy. Wenn $\lambda$ sehr groß wird, nähert sich $\exp(A/\lambda)$ der 1 an, und die Policy wird der Behavior Policy $\pi^D$ ähnlicher (stärkeres Constraint).

**A19:** Information-Directed Sampling wählt $\arg\min_a \frac{\Delta(a)^2}{g(a)}$:
- Aktion 1: $\frac{0.5^2}{0.25} = \frac{0.25}{0.25} = 1.0$
- Aktion 2: $\frac{0.3^2}{0.1} = \frac{0.09}{0.1} = 0.9$

Aktion 2 wird gewählt (niedrigerer Wert = besseres Verhältnis von Suboptimalität zu Information Gain).

**A20:** DPO Loss:
$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}\left[\log \sigma\left(\beta \cdot \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \cdot \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

Unterschied zu RLHF: DPO benötigt kein separates Reward-Modell und kein RL-Training (PPO). Es optimiert das LLM direkt auf den Präferenzen.

### Teil C Antworten

**A21:**

| Aspekt | UCB | Thompson Sampling | Information-Directed Sampling |
|--------|-----|-------------------|-------------------------------|
| Grundidee | Optimismus bei Unsicherheit | Sample aus Posterior | Balance Suboptimalität/Info |
| Bayesianisch? | Nein | Ja | Ja |
| Explorations-Bonus | $\sqrt{\frac{2\ln T}{N(a)}}$ | Implizit durch Sampling | $\Delta(a)^2/g(a)$ |
| Regret-Bound | $O(\log T)$ | $O(\log T)$ | $O(\log T)$ |

**A22:** Online RL: Agent interagiert mit Umgebung und sammelt eigene Daten. Vorteile: Fehler können korrigiert werden, Exploration möglich. Nachteile: Teuer, riskant, viele Interaktionen nötig.

Offline RL: Lernen aus festem Datensatz ohne Interaktion. Vorteile: Sicher, günstig, Daten wiederverwendbar. Nachteile: Distribution Shift, keine Fehlerkorrektur, OOD-Problem.

**A23:** Behavioral Cloning: Nur Experten-Daten, $O(T^2)$ Regret, einfach. DAgger: Iterativ eigene Daten + Experten-Labels, $O(T)$ Regret, aber Experte muss online verfügbar sein und viele Zustände labeln.

**A24:** Beide lösen das OOD-Problem in Offline RL. CQL: Reduziert Q-Werte für OOD-Aktionen explizit (konservative Schätzung). IQL: Verwendet Expectile Loss für Value-Funktion, vermeidet OOD-Aktionen durch asymmetrische Fehlerbewertung. IQL ist einfacher zu implementieren (kein zusätzlicher Regularizer nötig).

**A25:** Policy Constraint: Explizite Constraints auf Policy-Distanz (KL oder Support). Implicit Policy Constraints: Gewichtetes Behavioral Cloning mit Advantage. CQL: Konservative Q-Wert-Schätzung durch Regularisierung. IQL: Expectile Loss für Value-Funktion, implizite Policy-Verbesserung.

**A26:** DT ist return-konditioniertes Behavioral Cloning - es generiert Aktionen direkt ohne Planung. TT ist ein generatives Weltmodell, das Trajektorien vorhersagt und mit Beam Search die beste Folge sucht. Beam Search ermöglicht Trajectory Stitching (Kombination guter Teile verschiedener Trajektorien).

**A27:** RLHF benötigt: Reward-Modell, RL-Algorithmus (PPO), separate Training-Phasen. DPO benötigt nur: Das LLM selbst, Referenz-Modell. DPO-Vorteil: Einfacher, kein komplexes RL. DPO-Nachteil: Kann bei sehr großen Unterschieden zwischen $\pi_\theta$ und $\pi_{ref}$ instabil werden.

**A28:** Chain-of-Thought: Schritt-für-Schritt Denken in natürlicher Sprache. Tree of Thoughts: Mehrere Gedankenpfade parallel erkunden, Backtracking möglich. Self-Consistency: Mehrere Samples generieren, Mehrheitsentscheidung. CoT für einfache Reasoning, ToT für komplexe Probleme mit Verzweigungen, Self-Consistency für Robustheit.

**A29:** RAG kombiniert LLM mit externer Wissensdatenbank. Ablauf: Query → Embedding → Ähnlichkeitssuche → Kontext + Query → LLM → Antwort. Vorteile: Aktuelle Informationen, Quellenangaben, reduzierte Halluzinationen, kein Fine-Tuning nötig.

**A30:** Constitutional AI verwendet eine "Verfassung" (Regeln) statt menschlichen Feedbacks. SL-CAI: Self-Critique und Revision mit anschließendem Supervised Finetuning. RL-CAI: RL from AI Feedback (RLAIF) mit einem auf AI-Rankings trainierten Preference Model.

### Teil D Antworten

**A31:** Montezuma's Revenge ist schwierig wegen sparse Rewards und langen Zeithorizonten. Der Agent muss komplexe Sequenzen lernen (Schlüssel aufsammeln → Tür öffnen), ohne direktes Feedback. Empfohlene Methode: Pseudo-Counts oder RND für Neuigkeits-basierte Exploration, da sie Zustände erkunden, die potenziell zu neuen Entdeckungen führen.

**A32:** BC scheitert wegen Distributional Shift - die Policy macht Fehler und kommt in Zustände, die nicht in den Trainingsdaten sind. Lösungen: 1) Data Augmentation mit simulierten Fehlern (NVIDIA-Ansatz), 2) DAgger für iterative Datensammlung, 3) Goal-Conditioned BC für breitere Zustandsabdeckung.

**A33:** Im Offline RL kann die Policy keine neuen Aktionen ausprobieren, um Feedback zu erhalten. Wenn die Q-Funktion einen zu hohen Wert für eine OOD-Aktion vorhersagt, wird diese gewählt, aber der Fehler kann nie korrigiert werden (keine Umgebungsinteraktion). Fehler akkumulieren sich über die Zeit.

**A34:** Reward Hacking: Das Modell findet Schlupflöcher im Reward-Modell und maximiert den Reward auf unerwünschte Weise (z.B. Toxizität, Manipulation). Verhinderung: KL-Divergenz Constraint zur ursprünglichen Policy, sorgfältiges Reward-Modell Design, menschliche Überwachung.

**A35:** Der Bonus nimmt mit $N(a)$ ab, weil wir bei mehr Besuchen sicherer über die Qualität der Aktion sind (weniger Unsicherheit). Der Bonus steigt mit $\ln(T)$, weil bei späteren Zeitpunkten die Kosten von Exploration sinken (noch mehr Zeit für Exploitation verbleibt).

**A36:** Wenn der Posterior eng wird, konzentriert sich Thompson Sampling auf die wahrscheinlich beste Aktion (fast reine Exploitation). UCB exploriert weiterhin systematisch, bis die Unsicherheit unter einen Schwellenwert fällt.

**A37:** Bei MSE würde $V(s)$ den Durchschnitt von $Q(s,a)$ unter $\pi^D$ approximieren. Mit Expectile Loss und $\tau > 0.5$ approximiert $V(s)$ den Wert der besten durch die Daten gestützten Policy (oberer Expectile), was eine implizite Policy-Verbesserung ermöglicht.

**A38:** Die verbesserte CQL-Schranke garantiert, dass nur der Erwartungswert unter der Policy konservativ ist, nicht jedes einzelne Q(s,a). Dies verhindert extreme Pessimismus, während immer noch OOD-Aktionen bestraft werden.

**A39:** Exposure Bias: Das Modell wird mit echten Daten trainiert (Teacher Forcing), aber bei Inferenz mit eigenen Fehlern konfrontiert. Ähnlich zu Distributional Shift bei IL - die Test-Verteilung unterscheidet sich von der Trainings-Verteilung.

**A40:** Offline RL kann Trajectory Stitching: Gute Teile verschiedener Trajektorien können kombiniert werden (z.B. Trajektorie 1: A→B→C, Trajektorie 2: C→D→E → Offline RL lernt A→B→C→D→E). BC kann nur komplette Trajektorien kopieren. Theoretisch bewiesen unter bestimmten Annahmen.

</details>

---

## Bewertungsschlüssel

| Richtige Antworten | Bewertung |
|-------------------|-----------|
| 36-40 | 🟢 Exzellent - Bereit für die Prüfung |
| 30-35 | 🟢 Gut - Kleine Wiederholung empfohlen |
| 24-29 | 🟡 Befriedigend - Themen wiederholen |
| 18-23 | 🟡 Ausreichend - Tag 6 wiederholen |
| <18 | 🔴 Nachholbedarf - Zusammenfassungen nochmal lesen |

---

## Wichtige Formeln (auswendig lernen!)

**UCB:**
```
a = argmax_a [μ̂_a + √(2 ln(T) / N(a))]
```

**Regret:**
```
Reg(T) = T · E[r(a*)] - Σ E[r(a_t)]
```

**Pseudo-Count:**
```
N̂(s) = p_θ(s) / (p_θ'(s) - p_θ(s))
```

**Information Gain:**
```
IG(z, y|a) = E_y[H(p̂(z)) - H(p̂(z|y))|a]
```

**CQL Objective:**
```
L_CQL(Q) = α E_{s,a~μ}[Q(s,a)] - α E_{(s,a)~D}[Q(s,a)] + L_TD(Q)
```

**Expectile Loss:**
```
l_τ(x) = {(1-τ)|x|² if x > 0; τ|x|² if x ≤ 0}
```

**Advantage-Weighted Policy:**
```
π*(a|s) = (1/Z(s)) πᴰ(a|s) exp(A(s,a)/λ)
```

**DPO Loss:**
```
L_DPO = -E[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

**Information-Directed Sampling:**
```
a = argmin_a Δ(a)² / g(a)
```

---

**Viel Erfolg!** 🎯
