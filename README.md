
# Lift CV – Analisi Squat

---

## Obiettivo
Analizzare automaticamente **una singola ripetizione di squat** da video laterale.  
Fornisce validità della ripetizione e metriche quantitative come ROM, velocità, deviazione della barra e stima %1RM.

---

## Input
- Video laterale dello squat (corpo e barra visibili)  
- Peso sul bilanciere (opzionale)  
- Peso corporeo / storico 1RM (opzionali)

---

## Output
- Validità della ripetizione  
- Metriche quantitative  
- Report sintetico con limiti dichiarati

---

## Vincoli
- Analisi di **una sola ripetizione**  
- Solo squat, monocamera laterale  
- Nessun feedback in tempo reale  
- Alcune metriche non osservabili (es. valgismo ginocchio)

---

## Stack tecnico
Python + MediaPipe Pose + OpenCV + NumPy/SciPy

---

## Roadmap futura
- Panca piana  
- Serie multiple e trend %1RM  
- Stacco e multi-camera