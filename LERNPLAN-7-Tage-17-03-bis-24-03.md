# 7-Tage-Lernplan: Advanced ML (17.03. - 24.03.2026)

**Klausurdatum:** 24.03.2026  
**Heute:** Dienstag, 17.03.2026 (Dozenten-Termin war heute)  
**Verbleibend:** 7 Tage

---

## TAG 1 - DIENSTAG 17.03.
**Fokus:** Transformers & Self-Attention (🔴 SEHR WICHTIG)  
**Ziel:** Grundkonzept von Self-Attention, Q/K/V, Multi-Head verstehen

### Morgens (2.5 Stunden)
- [ ] 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md lesen (Seiten 1-25)
- [ ] Self-Attention Formel verstehen: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- [ ] Encoder vs Decoder Architektur skizzieren
- [ ] Positional Encoding Zweck notieren (nicht Formel auswendig!)
- [ ] 📺 Video: "Attention in transformers, visually explained" - 3Blue1Brown - https://www.youtube.com/watch?v=eMlx5fFNoYc (~20 Min)

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen beantworten:
  - Was ist Self-Attention? Erklären Sie Q, K, V!
  - Warum braucht man Positional Encoding?
  - Was ist der Unterschied zwischen Encoder und Decoder?
  - Wie funktioniert Multi-Head Attention?
  - Was ist Masked Attention?
- [ ] Dozenten-Notizen von heute einarbeiten

### Abends (1 Stunde)
- [ ] Kurze Wiederholung der 3 Kernpunkte
- [ ] Offene Fragen für morgen notieren

**Selbsttest:** Kannst du Self-Attention mit eigenen Worten erklären?

---

## TAG 2 - MITTWOCH 18.03.
**Fokus:** LSTM & Word Embeddings (🔴 SEHR WICHTIG)  
**Ziel:** LSTM Gates, Word2Vec, Bahdanau-Attention verstehen

### Morgens (2.5 Stunden)
- [ ] 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md lesen
- [ ] LSTM 3 Gates zeichnen (Forget, Input, Output)
- [ ] Cell State Update nachvollziehen
- [ ] Word2Vec (CBOW vs Skip-gram) Unterschiede notieren
- [ ] Bahdanau-Attention für Seq2Seq verstehen
- [ ] 📺 Video: "Word Embedding and Word2Vec, Clearly Explained!!!" - StatQuest - https://www.youtube.com/watch?v=viZrOnJclY0 (~17 Min)
- [ ] 📺 Video: "fastText Tutorial" - CodeEmporium - https://www.youtube.com/watch?v=Br-Ozg9D4mc (~12 Min)
- [ ] 📺 Video: "Byte Pair Encoding - How does the BPE algorithm work?" - https://www.youtube.com/watch?v=BcxJk4WQVIw (~10 Min)

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen:
  - Erklären Sie die 3 Gates im LSTM!
  - Was ist der Unterschied zwischen CBOW und Skip-gram?
  - Was ist die Distributional Hypothesis?
  - Wie funktioniert Bahdanau-Attention?
  - Vanishing Gradients Problem bei RNNs?
- [ ] Dozenten-Frage zu LSTM Parametern beantworten

### Abends (1 Stunde)
- [ ] Transformer vs LSTM Vergleich schreiben
- [ ] Leichte Wiederholung der Attention-Mechanismen

**Selbsttest:** Kannst du LSTM-Gates aufzeichnen und erklären?

---

## TAG 3 - DONNERSTAG 19.03.
**Fokus:** Q-Learning & Double DQN (🔴 SEHR WICHTIG)  
**Ziel:** Q-Learning Update, Bellman-Gleichung, Double DQN Problem

### Morgens (2.5 Stunden)
- [ ] 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md lesen (Seiten 1-40)
- [ ] Q-Learning Update-Formel auswendig: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
- [ ] Bellman-Optimalitätsgleichung verstehen
- [ ] Double DQN: Overestimation Problem notieren
- [ ] Experience Replay & Target Networks Zweck
- [ ] 📺 Video: "Bellman Equation - Explained!" - CodeEmporium - https://www.youtube.com/watch?v=9JZID-h6ZJ0 (~15 Min)
- [ ] 📺 Video: "The Power of Q-Learning in AI" - SuperDataScience - https://www.youtube.com/watch?v=_Dkf-7Oc6YQ (~18 Min)

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen:
  - Was ist der Unterschied zwischen Q-Learning und SARSA?
  - Was ist das Overestimation Problem?
  - Wie löst Double DQN das Problem?
  - Wozu dienen Target Networks?
  - Was ist Experience Replay?
- [ ] Dozenten-Frage zu Double DQN beantworten

### Abends (1 Stunde)
- [ ] Kurze Formel-Wiederholung
- [ ] ε-greedy vs UCB Konzept überblicken

**Selbsttest:** Kannst du Q-Learning Update ohne Nachschauen schreiben?

---

## TAG 4 - FREITAG 20.03.
**Fokus:** GANs & VAEs (🔴🟡 WICHTIG)  
**Ziel:** Generator/Discriminator, Minimax Loss, ELBO, Reparametrisierung

### Morgens (2.5 Stunden)
- [ ] 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md lesen (Seiten 1-30)
- [ ] GAN Grundkonzept: Generator & Discriminator
- [ ] Minimax Loss verstehen (nicht auswendig!)
- [ ] Mode Collapse Problem notieren
- [ ] VAE: Reparametrisierungs-Trick, ELBO Loss
- [ ] 📺 Video: "What are Generative Models? VAE & GAN" - Cambridge PhD - https://www.youtube.com/watch?v=24oBV_s5ufQ (~12 Min)

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen:
  - Was ist Mode Collapse bei GANs?
  - Erklären Sie Generator vs Discriminator!
  - Was ist die ELBO beim VAE?
  - Warum kann Autoencoder nicht sinnvoll generieren?
  - Was ist der Reparametrisierungs-Trick?
- [ ] Conditional GAN Konzept überblicken

### Abends (1 Stunde)
- [ ] GAN vs VAE Vergleich (Tabelle)
- [ ] Leichte Wiederholung der Loss-Funktionen

**Selbsttest:** Kannst du den VAE Reparametrisierungs-Trick erklären?

---

## TAG 5 - SAMSTAG 21.03.
**Fokus:** Diffusion & XAI (🟡 WICHTIG)  
**Ziel:** Forward/Reverse Process, LIME, SHAP Grundideen

### Morgens (2.5 Stunden)
- [ ] 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md (Seiten 31-60)
- [ ] Diffusion: Forward Process (Noise hinzufügen)
- [ ] Reverse Process (Denoising)
- [ ] Training Loss Konzept
- [ ] 06_ZUSAMMENFASSUNG-04-XAI.md lesen (Seiten 1-25)
- [ ] LIME: lokale Approximation
- [ ] SHAP: Shapley Values Grundidee
- [ ] 📺 Video: "Understanding Diffusion Models: Step-by-Step Explanation" - https://www.youtube.com/watch?v=0bRX0FNsRao
- [ ] 📺 Video: "Explainable AI Made Easy: SHAP, LIME & PFI" - https://www.youtube.com/watch?v=YZwDizPBFaM (~20 Min)

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen:
  - Erklären Sie Forward und Reverse Process bei Diffusion!
  - Was ist Classifier-Free Guidance?
  - Wie funktioniert LIME?
  - Was sind Shapley Values?
  - Vor-/Nachteile der generativen Modelle?
- [ ] Permutation Feature Importance überblicken

### Abends (1 Stunde)
- [ ] Diffusion vs GAN vs VAE Vergleich
- [ ] XAI Methoden Liste durchgehen

**Selbsttest:** Kannst du den Diffusion-Prozess in 3 Sätzen erklären?

---

## TAG 6 - SONNTAG 22.03.
**Fokus:** RL Exploration & Imitation Learning (🟡 WICHTIG)  
**Ziel:** UCB, Thompson Sampling, DAgger, Behavioral Cloning

### Morgens (2.5 Stunden)
- [ ] 10_ZUSAMMENFASSUNG-08-RL-Teil-2.md lesen (Seiten 1-35)
- [ ] UCB Formel: a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]
- [ ] Thompson Sampling Konzept
- [ ] Exploration vs Exploitation
- [ ] 08_ZUSAMMENFASSUNG-06-IL.md lesen
- [ ] Behavioral Cloning
- [ ] Distributional Shift Problem
- [ ] DAgger Algorithmus
- [ ] 📺 Video: "Upper Confidence Bound vs Thompson Sampling" - https://www.youtube.com/watch?v=e4f0or7x5xc (~10 Min)
- [ ] 📺 Video: "Behavior Cloning (Part 3): DAgger" - https://www.youtube.com/watch?v=gffZio5UmQc (~8 Min)

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen:
  - Wie funktioniert UCB?
  - Was ist Thompson Sampling?
  - Was ist das Problem bei Offline RL?
  - Was ist Distributional Shift beim Imitation Learning?
  - Wie funktioniert DAgger?
- [ ] Offline RL Schwierigkeit notieren

### Abends (1 Stunde)
- [ ] On-Policy vs Off-Policy Wiederholung
- [ ] RLHF & DPO Grundideen überblicken (05_ZUSAMMENFASSUNG-03)

**Selbsttest:** Kannst du UCB-Formel schreiben und erklären?

---

## TAG 7 - MONTAG 23.03.
**Fokus:** PUFFERTAG - Wiederholung & Schwachstellen  
**Ziel:** Alle Themen überblicken, offene Fragen klären

### Morgens (2 Stunden)
- [ ] ALLE Zusammenfassungen überfliegen (03-10)
- [ ] Eigene Notizen durchgehen
- [ ] Schwache Themen identifizieren
- [ ] 30 Selbsttest-Fragen aus PROJECT-File beantworten
- [ ] Offene Fragen klären (Fragenkatalog)

### Nachmittags (2 Stunden)
- [ ] Schwache Themen gezielt wiederholen
- [ ] Wichtige Formeln wiederholen:
  - Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
  - Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
  - UCB: a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]
- [ ] Dozenten-Fragen (15 Fragen) komplett durchgehen

### Abends (1 Stunde)
- [ ] LEICHTER Überblick - nicht mehr neu lernen!
- [ ] Früh ins Bett gehen!
- [ ] Materialien für morgen bereitlegen

**Selbsttest:** Fühlst du dich sicher bei allen 🔴 Themen?

---

## TAG 8 - DIENSTAG 24.03. - KLAUSURTAG 📝

### Morgens (1 Stunde vor Klausur)
- [ ] Gutes Frühstück!
- [ ] NUR kurze Wiederholung:
  - Self-Attention Konzept (1 Minute)
  - LSTM Gates (1 Minute)
  - Q-Learning Formel (1 Minute)
  - GAN Grundprinzip (1 Minute)
- [ ] NICHTS NEUES MEHR!
- [ ] Zur Klausurteilnahme denken!

### In der Klausur
- [ ] Erst einfache Fragen lösen
- [ ] Bei Unsicherheit: Markieren und später zurückkommen
- [ ] Zeitmanagement: Nicht zu lange an einer Frage hängen
- [ ] Konzepte erklären, nicht alles auswendig!

**Erfolg!** 🎯

---

## 📊 Priorisierungsübersicht

| Priorität | Themen | Tage |
|-----------|--------|------|
| 🔴 SEHR WICHTIG | Transformers, LSTM, Q-Learning, GANs, Word Embeddings | Tag 1-4 |
| 🟡 WICHTIG | VAE, Diffusion, XAI, RL Exploration, Imitation Learning | Tag 4-6 |
| 🟢 GRUNDWISSEN | RLHF, DPO, Offline RL, XAI intrinsisch | Tag 6-7 (wenn Zeit) |

---

## 📝 Wichtige Formeln (auswendig)

1. **Self-Attention:** Attention(Q,K,V) = softmax(QK^T/√d_k)V
2. **Q-Learning:** Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
3. **UCB:** a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]

---

## ⚠️ Nicht klausurrelevant (NICHT lernen!)

❌ Tokenisierung Details (BPE etc.)  
❌ Aktivierungsfunktionen Details  
❌ Positional Encodings Formeln  
❌ Metriken & Benchmarks  
❌ Layer-wise relevance propagation  
❌ Integrated Gradients  
❌ Wasserstein GAN  
❌ DDPG, TRPO, PPO, SAC  
❌ Implementierungstipps  

---

**Viel Erfolg!** 🍀
