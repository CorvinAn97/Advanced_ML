# Selbsttest Tag 3: Reinforcement Learning Teil 1

**Umfang:** Ausführlicher Test zu allen Themen des Tages  
**Zeitansatz:** 50-70 Minuten  
**Hinweis:** Antworten nicht einfach ablesen - erst selbst überlegen, dann nachschlagen!

---

## Teil A: Grundkonzepte & Verständnis (12 Fragen)

### 1. RL-Grundlagen

**Frage 1:**  
Erklären Sie die Interaktion zwischen Agent und Environment. Zeichnen Sie das Standard-Diagramm mit State, Action, Reward und dem zeitlichen Ablauf.

**Frage 2:**  
Was bedeutet die **Reward-Hypothese** im Reinforcement Learning? Warum ist diese Annahme fundamental für RL-Algorithmen?

**Frage 3:**  
Was ist der Unterschied zwischen episodischen Tasks und continuing tasks? Geben Sie jeweils ein konkretes Beispiel.

---

### 2. Policy & Return

**Frage 4:**  
Unterscheiden Sie zwischen deterministischer und stochastischer Policy. Schreiben Sie die mathematische Notation für beide Fälle auf.

**Frage 5:**  
Was ist der Return G_t? Schreiben Sie die Formel auf und erklären Sie die Rolle des Discount-Faktors γ. Was passiert bei γ = 0 vs. γ = 0.99?

**Frage 6:**  
Warum ist der Discount-Faktor γ notwendig? Was würde passieren, wenn γ = 1 in einem continuing task (unendlicher Horizont) verwendet würde?

---

### 3. Value Functions

**Frage 7:**  
Was ist der Unterschied zwischen der State-Value Function v_π(s) und der Action-Value Function q_π(s,a)? Welche Information enthält q_π, die v_π nicht hat?

**Frage 8:**  
Wie berechnet sich v_π(s) aus q_π(s,a)? Schreiben Sie die Formel auf und erklären Sie, warum eine Erwartung über die Policy notwendig ist.

**Frage 9:**  
Was sind die optimalen Value Functions v*(s) und q*(s,a)? Warum ist die Kenntnis von q*(s,a) besonders wertvoll für die Bestimmung der optimalen Policy?

---

### 4. Exploration vs. Exploitation

**Frage 10:**  
Erklären Sie das Dilemma zwischen Exploration und Exploitation. Warum ist beides notwendig und was passiert bei zu viel von jeweils einem?

**Frage 11:**  
Wie funktioniert ε-Greedy Exploration? Schreiben Sie die Policy-Definition auf und erklären Sie, was bei ε = 0.1 passiert.

**Frage 12:**  
Warum sollte ε im Laufe des Trainings typischerweise reduziert werden (ε-Decay)? Was ist das Ziel?

---

## Teil B: Formeln & Berechnungen (10 Fragen)

### 5. Bellman-Gleichungen

**Frage 13:**  
Schreiben Sie die **Bellman-Expectation-Gleichung** für v_π(s) aus dem Gedächtnis auf. Erklären Sie jeden Term.

**Frage 14:**  
Schreiben Sie die **Bellman-Optimality-Gleichung** für q*(s,a) auf. Warum ist diese Gleichung die Grundlage für Q-Learning?

**Frage 15:**  
Gegeben: Ein Agent befindet sich in Zustand s. Es gibt 2 Aktionen mit folgenden Q-Werten: Q(s,a₁) = 10, Q(s,a₂) = 8. Die Policy π wählt a₁ mit 70% und a₂ mit 30% Wahrscheinlichkeit.
- Berechnen Sie v_π(s).
- Was wäre v*(s)?

---

### 6. TD Learning Updates

**Frage 16:**  
Schreiben Sie das **TD(0) Update** für V(S_t) auf. Was ist der TD Target und was ist der TD Error?

**Frage 17:**  
Ein Agent erhält: State S_t, führt Aktion aus, bekommt Reward R_{t+1} = 5, gelangt zu S_{t+1}. Die aktuellen Schätzungen sind V(S_t) = 10, V(S_{t+1}) = 15. γ = 0.9, α = 0.1.
- Berechnen Sie den TD Target.
- Berechnen Sie den TD Error.
- Berechnen Sie das neue V(S_t).

**Frage 18:**  
Schreiben Sie das **SARSA Update** und das **Q-Learning Update** nebeneinander auf. Markieren Sie den entscheidenden Unterschied.

---

### 7. Parameter & Komplexität

**Frage 19:**  
Ein Q-Table hat |S| = 1000 Zustände und |A| = 5 Aktionen. Wie viele Einträge hat die Tabelle? Warum ist das problematisch für Bild-Eingaben mit 84×84 Pixeln und 256 Grauwerten?

**Frage 20:**  
Berechnen Sie den Skalierungsfaktor für die TD Error-Berechnung: Wenn α = 0.1 und der TD Error δ = 2.5, um wie viel wird Q(s,a) aktualisiert?

**Frage 21:**  
Ein Agent sammelt folgende Episode: s₀→a₀→r₁=1→s₁→a₁→r₂=2→s₂→a₂→r₃=3→s₃ (terminal). Berechnen Sie G₀, G₁, G₂ für γ = 0.9.

**Frage 22:**  
Gegeben: Ein DQN-Netzwerk hat Input-Dimension 4 (State) und Output-Dimension 3 (Aktionen). Die erste Hidden Layer hat 64 Neuronen. Wie viele Parameter hat die erste Schicht (Input → Hidden)?

---

## Teil C: Vergleiche & Analyse (10 Fragen)

### 8. MC vs. TD

**Frage 23:**  
Vergleichen Sie Monte-Carlo (MC) und Temporal-Difference (TD) Learning in einer Tabelle:

| Aspekt | Monte-Carlo | TD(0) |
|--------|-------------|-------|
| Target | ? | ? |
| Bias | ? | ? |
| Varianz | ? | ? |
| Bootstrapping | ? | ? |
| Update-Zeitpunkt | ? | ? |

**Frage 24:**  
Warum hat MC keine Bias, aber hohe Varianz? Warum hat TD niedrigere Varianz, aber etwas Bias?

**Frage 25:**  
In welchen Situationen würden Sie MC bevorzugen? In welchen TD? Begründen Sie.

---

### 9. SARSA vs. Q-Learning

**Frage 26:**  
Erklären Sie den fundamentalen Unterschied zwischen SARSA (On-Policy) und Q-Learning (Off-Policy). Welche Aktion wird bei jedem Update verwendet?

**Frage 27:**  
Das "Cliff Walking" Beispiel: SARSA lernt einen sicheren Pfad weg vom Cliff, Q-Learning lernt den optimalen Pfad entlang des Cliffs. Erklären Sie warum.

**Frage 28:**  
Warum kann Q-Learning aus "fremden" Daten lernen (z.B. aus Experience Replay), während SARSA das nicht kann?

---

### 10. DQN & Double DQN

**Frage 29:**  
Was ist das **Overestimation Problem** bei DQN? Warum tritt es auf und wie entsteht die systematische Überschätzung?

**Frage 30:**  
Wie löst Double DQN das Overestimation Problem? Erklären Sie die Entkopplung von Selektion und Bewertung.

**Frage 31:**  
Vergleichen Sie DQN und Double DQN:

| Aspekt | DQN | Double DQN |
|--------|-----|------------|
| Target Berechnung | ? | ? |
| Overestimation | ? | ? |
| Performance | ? | ? |
| Implementierungsaufwand | ? | ? |

**Frage 32:**  
Warum sind Target Networks wichtig für DQN? Was wäre das "Moving Target Problem" ohne sie?

---

## Teil D: Praktische Anwendungen & Edge Cases (8 Fragen)

### 11. Experience Replay & Training

**Frage 33:**  
Was sind die drei Hauptvorteile von Experience Replay? Warum wird zufällig gesampelt statt chronologisch?

**Frage 34:**  
Ein Replay Buffer hat Kapazität N = 100.000. Wie groß ist der Speicherbedarf pro Transition (s, a, r, s'), wenn States als Float32-Vektoren der Länge 4 gespeichert werden? Berechnen Sie den Gesamtspeicherbedarf in MB.

---

### 12. On-Policy vs. Off-Policy

**Frage 35:**  
Erklären Sie den Unterschied zwischen Target Policy und Behavior Policy bei Off-Policy Methoden. Warum ist diese Trennung vorteilhaft?

**Frage 36:**  
Warum ist Q-Learning als Off-Policy Methode effizienter in der Datennutzung als SARSA? Was bedeutet "Sample Efficiency"?

---

### 13. Function Approximation

**Frage 37:**  
Warum ist Function Approximation mit neuronalen Netzen notwendig für DQN? Was ist das Problem mit Q-Tables bei Atari-Spielen?

**Frage 38:**  
Was sind Semi-Gradient Methods? Warum wird der Gradient als "semi" bezeichnet, wenn TD Target verwendet wird?

---

### 14. Policy Gradients & Actor-Critic (Grundlagen)

**Frage 39:**  
Was ist die Grundidee von Policy Gradient Methoden (REINFORCE)? Wie unterscheiden sie sich von Value-Based Methods wie Q-Learning?

**Frage 40:**  
Was ist die Actor-Critic Architektur? Welche Rolle spielen Actor und Critic jeweils?

---

## Antworten & Lösungen

<details>
<summary>Klicken Sie hier, um die Antworten anzuzeigen</summary>

### Teil A Antworten

**A1:** Agent-Environment Interaktion:
```
State s_t → Agent wählt Action a_t → Environment → Reward r_{t+1}, Next State s_{t+1}
```
Zeitlicher Ablauf: S₀, A₀, R₁, S₁, A₁, R₂, S₂, ... (Markov Decision Process)

**A2:** Reward-Hypothese: "Alles, was wir als Ziel betrachten, kann als Maximierung eines erwarteten kumulativen Rewards formalisiert werden." Fundamental, weil RL-Algorithmen einheitlich auf Reward-Maximierung optimieren können, unabhängig vom konkreten Ziel.

**A3:** Episodisch: Episode endet bei terminalem Zustand (z.B. Schachspiel, Level in einem Spiel). Continuing: Keine terminalen Zustände, unendlicher Horizont (z.B. Roboter-Steuerung, Aktienhandel).

**A4:** Deterministisch: a = π(s) - eine feste Aktion pro Zustand. Stochastisch: a ~ π(·|s) - Wahrscheinlichkeitsverteilung über Aktionen, π(a|s) gibt Wahrscheinlichkeit für Aktion a in Zustand s.

**A5:** Return G_t = Σₖ₌₀^∞ γᵏ · rₜ₊ₖ₊₁. γ ∈ [0,1] ist der Discount-Faktor. γ = 0: Nur sofortiger Reward zählt (myopisch). γ = 0.99: Langfristige Rewards fast gleich gewichtet wie kurzfristige.

**A6:** γ notwendig, um unendliche Summen zu begrenzen und zukünftige Rewards zu diskontieren. Bei γ = 1 in continuing tasks würde G_t unendlich werden (keine Konvergenz).

**A7:** v_π(s) = erwarteter Return ab Zustand s unter Policy π. q_π(s,a) = erwarteter Return bei Ausführung von Aktion a in Zustand s, dann Fortsetzung mit π. q_π enthält die Information über die unmittelbare Konsequenz einer spezifischen Aktion.

**A8:** v_π(s) = Σₐ π(a|s) · q_π(s,a). Erwartung notwendig, weil Policy π stochastisch sein kann - wir müssen über alle möglichen Aktionen mitteln, gewichtet mit ihrer Wahrscheinlichkeit.

**A9:** v*(s) = max_π v_π(s), q*(s,a) = max_π q_π(s,a). Aus q* kann man direkt die optimale Policy berechnen: π*(s) = argmaxₐ q*(s,a).

**A10:** Exploration = neue Dinge ausprobieren, Exploitation = bekannte gute Aktionen nutzen. Zu viel Exploration → schlechte Performance, zu viel Exploitation → suboptimale Policy, lokale Optima.

**A11:** ε-Greedy: π(a|s) = 1-ε+ε/|A| für beste Aktion, ε/|A| für andere. Bei ε=0.1: 90% Greedy, 10% zufällige Exploration.

**A12:** ε-Decay reduziert Exploration über Zeit. Ziel: Am Anfang viel erkunden, später ausnutzen des Gelernten. Typisch: Start ε=1.0, Ende ε=0.01.

### Teil B Antworten

**A13:** Bellman-Expectation für v_π:
```
v_π(s) = Σₐ π(a|s) · Σₛ'ᵣ p(s',r|s,a) · [r + γ·v_π(s')]
```
Terme: Summe über Aktionen (Policy), Summe über Zustände/Rewards (Transition), sofortiger Reward + diskontierter zukünftiger Wert.

**A14:** Bellman-Optimality für q*:
```
q*(s,a) = Σₛ'ᵣ p(s',r|s,a) · [r + γ·maxₐ' q*(s',a')]
```
Grundlage für Q-Learning, weil Q-Learning direkt versucht, diese Gleichung zu erfüllen durch iteratives Update.

**A15:** v_π(s) = 0.7 × 10 + 0.3 × 8 = 7 + 2.4 = 9.4. v*(s) = max(10, 8) = 10 (unter optimaler Policy würde immer a₁ gewählt).

**A16:** TD(0) Update:
```
V(Sₜ) ← V(Sₜ) + α · [Rₜ₊₁ + γ·V(Sₜ₊₁) - V(Sₜ)]
```
TD Target: Rₜ₊₁ + γ·V(Sₜ₊₁), TD Error: Rₜ₊₁ + γ·V(Sₜ₊₁) - V(Sₜ)

**A17:** TD Target = 5 + 0.9 × 15 = 5 + 13.5 = 18.5. TD Error = 18.5 - 10 = 8.5. Neues V(Sₜ) = 10 + 0.1 × 8.5 = 10 + 0.85 = 10.85.

**A18:** SARSA: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)] (tatsächliche nächste Aktion). Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ·maxₐ' Q(s',a') - Q(s,a)] (beste nächste Aktion).

**A19:** 1000 × 5 = 5000 Einträge. Bei Bildern: 256^(84×84) ≈ 10^16984 Zustände - unmöglich in Tabelle zu speichern.

**A20:** ΔQ = α × δ = 0.1 × 2.5 = 0.25. Q(s,a) wird um 0.25 aktualisiert.

**A21:** G₂ = r₃ = 3. G₁ = r₂ + γ·G₂ = 2 + 0.9×3 = 2 + 2.7 = 4.7. G₀ = r₁ + γ·G₁ = 1 + 0.9×4.7 = 1 + 4.23 = 5.23.

**A22:** Input 4, Hidden 64: Parameter = (4 + 1) × 64 = 5 × 64 = 320 (4 Gewichte + 1 Bias pro Neuron).

### Teil C Antworten

**A23:**
| Aspekt | Monte-Carlo | TD(0) |
|--------|-------------|-------|
| Target | Gₜ (tatsächlicher Return) | Rₜ₊₁ + γ·V(Sₜ₊₁) |
| Bias | Kein Bias | Etwas Bias |
| Varianz | Hoch | Niedrig |
| Bootstrapping | Nein | Ja |
| Update-Zeitpunkt | Episode-Ende | Nach jedem Schritt |

**A24:** MC: Kein Bias, weil tatsächlicher Return verwendet wird. Hohe Varianz, weil viele zufällige Rewards summiert werden. TD: Niedrigere Varianz, weil nur ein zufälliger Reward. Etwas Bias, weil V(Sₜ₊₁) eine Schätzung ist (Bootstrapping).

**A25:** MC bevorzugen bei kurzen Episoden, wenn keine Markov-Eigenschaft gegeben ist, oder wenn Bias vermieden werden muss. TD bevorzugen bei langen Episoden, continuing tasks, oder wenn schnelles Online-Lernen wichtig ist.

**A26:** SARSA (On-Policy): Lernt Wert der aktuell verfolgten Policy, verwendet tatsächlich gewählte nächste Aktion a'. Q-Learning (Off-Policy): Lernt Wert der optimalen Policy, verwendet maxₐ' Q(s',a') unabhängig von tatsächlich gewählter Aktion.

**A27:** SARSA berücksichtigt Exploration in Q-Werten (auch "Fehler" durch zufällige Aktionen). Q-Learning ignoriert Exploration, lernt optimalen Pfad. SARSA lernt daher sicheren Pfad weg vom Cliff, Q-Learning riskiert Absturz durch Exploration.

**A28:** Q-Learning lernt optimale Policy unabhängig von verfolgter Policy. Kann daher aus beliebigen Daten lernen, solange alle (s,a) besucht werden. SARSA lernt nur die aktuelle Policy, daher inkonsistent mit alten Daten aus anderer Policy.

**A29:** Overestimation: DQN überschätzt systematisch Q-Werte. Ursache: max Operator auf noisy Schätzungen - max von Zufallsvariablen ist im Erwartungswert größer als max der wahren Werte (Maximization Bias).

**A30:** Double DQN entkoppelt: 1) Selektion mit Q-Network: a* = argmaxₐ Q(s',a;w). 2) Bewertung mit Target Network: Q_target(s',a*;w⁻). Verhindert, dass derselbe Wert für beides verwendet wird.

**A31:**
| Aspekt | DQN | Double DQN |
|--------|-----|------------|
| Target | r + γ·maxₐ' Q_target(s',a') | r + γ·Q_target(s',argmaxₐ' Q(s',a)) |
| Overestimation | Ja | Reduziert |
| Performance | Gut | Besser |
| Implementierung | Standard | Einfache Änderung |

**A32:** Target Networks werden nur periodisch aktualisiert. Ohne sie ändert sich das Target bei jedem Update → "Moving Target" → Instabilität. Mit Target Network bleibt Target stabil für C Schritte.

### Teil D Antworten

**A33:** Vorteile: 1) Bricht zeitliche Korrelationen (i.i.d. Samples), 2) Höhere Daten-Effizienz (jede Transition mehrfach verwendbar), 3) Glättet das Training. Zufälliges Sampling statt chronologisch, um Korrelationen zu brechen und i.i.d.-Annahme des SGD zu erfüllen.

**A34:** Pro Transition: s (4×4 Bytes) + a (4 Bytes) + r (4 Bytes) + s' (4×4 Bytes) = 16 + 4 + 4 + 16 = 40 Bytes. Gesamt: 100.000 × 40 = 4.000.000 Bytes ≈ 3.81 MB.

**A35:** Target Policy = die Policy, die wir lernen wollen (z.B. optimale Policy). Behavior Policy = die Policy, die wir tatsächlich ausführen (z.B. ε-greedy mit Exploration). Trennung ermöglicht Exploration während optimale Policy gelernt wird.

**A36:** Q-Learning kann aus Experience Replay lernen - jede Transition kann mehrfach verwendet werden, auch aus vergangenen Policies. SARSA muss frische Samples von aktueller Policy haben. Sample Efficiency = wie viele Samples nötig, um gute Policy zu lernen.

**A37:** Atari: 210×160×3 Pixel × 256 Werte = riesiger Zustandsraum. Function Approximation mit CNN generalisiert von gesehenen zu ungesehenen Zuständen. Q-Table unmöglich (mehr Zustände als Atome im Universum).

**A38:** Semi-Gradient: TD Target hängt von Parametern w ab (durch V(Sₜ₊₁;w)), aber diese Abhängigkeit wird im Gradienten ignoriert. Daher "semi" - nicht der volle Gradient, aber funktioniert in der Praxis.

**A39:** Policy Gradients optimieren direkt die Policy π(a|s;θ) durch Gradient Ascent auf dem erwarteten Return. Value-Based: Lernen zuerst Value Function, dann ableiten Policy. Policy Gradients: Keine Value Function nötig, direkte Policy-Optimierung.

**A40:** Actor-Critic: Actor = Policy-Netzwerk (wählt Aktionen), Critic = Value-Netzwerk (bewertet Aktionen). Critic schätzt Advantage/Value, Actor aktualisiert Policy basierend auf Critic's Bewertung. Kombiniert Policy Gradient mit Value Function als Baseline.

</details>

---

## Bewertungsschlüssel

| Richtige Antworten | Bewertung |
|-------------------|-----------|
| 36-40 | 🟢 Exzellent - Bereit für Tag 4 |
| 30-35 | 🟢 Gut - Kleine Wiederholung empfohlen |
| 24-29 | 🟡 Befriedigend - Themen wiederholen |
| 18-23 | 🟡 Ausreichend - Tag 3 wiederholen |
| <18 | 🔴 Nachholbedarf - Zusammenfassung nochmal lesen |

---

## Wichtige Formeln (auswendig lernen!)

### Return
```
Gₜ = Σₖ₌₀^∞ γᵏ · rₜ₊ₖ₊₁
```

### Bellman-Expectation für v_π
```
v_π(s) = Σₐ π(a|s) · Σₛ'ᵣ p(s',r|s,a) · [r + γ·v_π(s')]
```

### Bellman-Optimality für q*
```
q*(s,a) = Σₛ'ᵣ p(s',r|s,a) · [r + γ·maxₐ' q*(s',a')]
```

### TD(0) Update
```
V(Sₜ) ← V(Sₜ) + α · [Rₜ₊₁ + γ·V(Sₜ₊₁) - V(Sₜ)]
```

### SARSA Update (On-Policy)
```
Q(s,a) ← Q(s,a) + α · [r + γ·Q(s',a') - Q(s,a)]
```

### Q-Learning Update (Off-Policy) ⭐
```
Q(s,a) ← Q(s,a) + α · [r + γ·maxₐ' Q(s',a') - Q(s,a)]
```

### ε-Greedy Policy
```
π(a|s) = { 1-ε + ε/|A|   wenn a = argmaxₐ' Q(s,a')
         { ε/|A|         sonst
```

### DQN Loss
```
L(w) = E[(r + γ·maxₐ' Q_target(s',a';w⁻) - Q(s,a;w))²]
```

### Double DQN Update ⭐
```
a* = argmaxₐ' Q(s',a';w)                    # Selektion mit Q-Network
Q_target = r + γ·Q_target(s',a*;w⁻)          # Bewertung mit Target Network
```

---

## Zusammenfassung der Kernpunkte

| Konzept | Beschreibung | Schlüsselformel |
|---------|-------------|-----------------|
| **Return** | Kumulativer diskontierter Reward | Gₜ = Σ γᵏ·rₜ₊ₖ₊₁ |
| **v_π(s)** | State-Value unter Policy π | E_π[Gₜ\|Sₜ=s] |
| **q_π(s,a)** | Action-Value unter Policy π | E_π[Gₜ\|Sₜ=s,Aₜ=a] |
| **MC** | Lernen aus tatsächlichen Returns | Kein Bias, hohe Varianz |
| **TD** | Bootstrapping mit aktuellen Schätzungen | Niedrige Varianz, etwas Bias |
| **SARSA** | On-Policy TD Control | Q ← Q + α[r + γ·Q(s',a') - Q] |
| **Q-Learning** | Off-Policy TD Control | Q ← Q + α[r + γ·max Q' - Q] |
| **DQN** | Deep Learning + Q-Learning | Experience Replay + Target Networks |
| **Double DQN** | Reduziert Overestimation | Trennt Selektion & Bewertung |

---

**Tipps:**
- Markieren Sie Fragen, bei denen Sie unsicher waren
- Schauen Sie sich die zugehörigen Zusammenfassungs-Abschnitte nochmal an
- Üben Sie die Formeln auswendig zu schreiben (besonders Q-Learning Update!)
- Erklären Sie schwierige Konzepte einem imaginären Lernpartner (Feynman-Technik)

**Viel Erfolg!** 🎯
