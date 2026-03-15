# Vorbereitung für Dozenten-Termin (Dienstag)

**Ziel:** Konzepte grob verstehen + intelligente Fragen stellen

---

## 📅 Tagesplan

### Sonntag (heute) - Transformers & RNNs

**Konzepte (nur grobes Verständnis!):**

1. **Warum Transformer statt RNN?**
   - RNN: Sequentiell, langsame Weitergabe von Info
   - Transformer: Parallel, direkte Verbindungen zwischen allen Wörtern
   - **Nicht auswendig lernen:** Nur verstehen dass Transformer schneller sind

2. **Self-Attention (Intuition)**
   - Jedes Wort schaut auf alle anderen Wörter
   - "Der" schaut auf "Koch", "servierte", "Pizza"
   - **Nicht auswendig lernen:** Nur das Konzept verstehen

3. **LSTM vs. normales RNN**
   - Normales RNN: Vergisst schnell (vanishing gradient)
   - LSTM: Hat "Gedächtnis" (Cell State)
   - **Nicht auswendig lernen:** Nur wann man LSTM braucht

**Fragen die dir aufkommen könnten (notieren!):**
- Wie genau werden die Attention-Scores berechnet?
- Was ist der Unterschied zwischen Encoder und Decoder?
- Wann nimmt man LSTM, wann Transformer?

---

### Montag - Reinforcement Learning

**Konzepte (nur grobes Verständnis!):**

1. **Q-Learning (Grundidee)**
   - Agent lernt: In Situation X ist Aktion Y gut
   - Q-Table speichert "Qualität" von Aktionen
   - **Nicht auswendig lernen:** Nur das Prinzip verstehen

2. **Double DQN (Warum? Was löst es?)**
   - Problem: Normales DQN überschätzt Q-Werte
   - Lösung: Zwei Netzwerke, einer wählt, einer bewertet
   - **Nicht auswendig lernen:** Nur das Problem verstehen

3. **Policy Gradients vs. Q-Learning**
   - Q-Learning: Lernt Werte von Aktionen
   - Policy Gradients: Lernt direkt Strategie (Policy)
   - **Nicht auswendig lernen:** Nur den Unterschied verstehen

**Fragen die dir aufkommen könnten (notieren!):**
- Wie genau funktioniert die Exploration vs. Exploitation?
- Was ist der Unterschied zwischen Actor und Critic?
- Wann nimmt man Q-Learning, wann Policy Gradients?

---

### Dienstag (vor dem Termin) - Schnell-Check

**30 Minuten vor dem Termin:**
- Deine notierten Fragen durchlesen
- PROJECT-Advanced_ML.md überfliegen
- 3-4 konkrete Fragen auswählen

---

## ❓ Intelligente Fragen für den Dozenten

### Kategorie 1: "Verständnis-Check" (zeigt Engagement)

1. **"Ich habe mir Transformers angeguckt - verstehe das Konzept der Self-Attention, aber bei der mathematischen Berechnung der Attention-Scores bin ich mir unsicher. Können Sie das kurz erläutern oder ist das zu detailliert für die Klausur?"**

2. **"Bei LSTM verstehe ich die Idee des Cell State, aber der Unterschied zwischen forget gate, input gate und output gate ist mir noch nicht ganz klar. Ist das ein wichtiges Klausurthema?"**

3. **"Ich habe Q-Learning und Policy Gradients verglichen - Q-Learning lernt Werte, Policy Gradients lernt direkt die Strategie. Aber wann wählt man welchen Ansatz? Gibt es da Faustregeln?"**

### Kategorie 2: "Typische Fehler" (zeigt Voraussicht)

4. **"Was sind typische Fehler, die Studierende in der Klausur machen? Gibt es Themen die oft falsch verstanden werden?"**

5. **"Bei GANs vs. VAEs vs. Diffusion - woran erkennt man in der Klausur, welches Modell gefragt ist? Worin unterscheiden sich die Anwendungsfälle?"**

6. **"Bei XAI (LIME, SHAP) - was wird erwartet? Nur die Grundidee oder auch mathematische Details?"**

### Kategorie 3: "Detailtiefe" (zeigt Realismus)

7. **"Wie viel Detailtiefe wird bei Transformers erwartet? Müssen wir die komplette Architektur skizzieren können oder reicht das Konzeptuelle?"**

8. **"Bei RL: Müssen wir die Update-Gleichungen für Q-Learning herleiten können oder reicht das Verständnis des Algorithmus?"**

9. **"Gibt es Themen die in den Vorlesungen vorkamen, aber in der Klausur nicht relevant sind?"**

### Kategorie 4: "Prüfungsformat" (zeigt Planung)

10. **"Wie sind die Aufgaben in der Klausur typischerweise strukturiert? Rechenaufgaben, Konzeptverständnis, oder Vergleiche?"**

11. **"Gibt es Multiple-Choice-Fragen oder nur offene Fragen?"**

12. **"Wie viel Zeit sollte man für die einzelnen Aufgaben einplanen?"**

---

## 💡 Tipps für das Gespräch

### ✅ Das solltest du tun:

- **Zeigen dass du angefangen hast:** "Ich habe mir die ersten Themen angeguckt..."
- **Konkrete Unsicherheiten nennen:** Nicht "Ich verstehe nichts", sondern "Bei X verstehe ich Y, aber Z ist unklar"
- **Mitschreiben:** Notiere die Antworten des Dozenten
- **Nachhaken:** "Könnten Sie das Beispiel nochmal erklären?"

### ❌ Das solltest du vermeiden:

- **Nichts vorbereitet haben:** "Ich habe noch nicht angefangen"
- **Zu selbstsicher wirken:** "Ich kann das alles schon"
- **Unspezifisch sein:** "Ich verstehe Transformers nicht"
- **Andere fragen lassen:** Du sollst Fragen stellen, nicht nur zuhören

---

## 📝 Deine Notizen (hier eintragen!)

### Sonntag - Notizen:
```
[Platz für deine Fragen/Verständnisprobleme]

1. 
2. 
3. 
```

### Montag - Notizen:
```
[Platz für deine Fragen/Verständnisprobleme]

1. 
2. 
3. 
```

### Dienstag - Ausgewählte Fragen (3-4 Stück):
```
[Die Fragen die du wirklich stellen willst]

1. 
2. 
3. 
4. 
```

---

## 🎯 Erinnerung

**Ziel bis Dienstag:**
- ✅ Grobes Verständnis von Transformers, LSTM, RL
- ✅ 3-4 konkrete Fragen vorbereitet
- ✅ Zeigen dass du dich bemühst
- ✅ Nicht zu viel vorwegnehmen

**Nach dem Termin:**
- Deine Notizen in PROJECT-Advanced_ML.md eintragen
- Lernplan anpassen basierend auf dem was der Dozent gesagt hat
