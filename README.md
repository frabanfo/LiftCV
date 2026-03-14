# LiftCV — Squat Video Analysis

LiftCV is a computer vision pipeline that analyzes a **single squat repetition** from a lateral video and extracts biomechanical metrics and validity checks.

The project is a **proof-of-concept tool for powerlifting analysis using computer vision**.

---

## Features

LiftCV evaluates a squat repetition and returns:

**Validity checks**

- Squat depth  
- Initial lockout  
- Final lockout  
- Foot stability  

**Kinematic metrics**

- Bar range of motion (ROM)  
- Mean concentric velocity  
- Peak velocity  
- Horizontal bar deviation  
- Estimated %1RM (velocity-based)

---

## Input

The system expects:

| Input | Description | Required |
|------|-------------|---------|
| Squat video | Lateral view with athlete and bar visible | mandatory |
| Barbell weight | Used for intensity estimation | mandatory |
| Athlete height | Improves scale normalization | mandatory |
| Bodyweight | Optional contextual metric | optional |
| Historical 1RM | Optional intensity reference | optional |

---

## Output

LiftCV produces a structured report including:

- Rep validity assessment  
- Computed kinematic metrics  
- Estimated training intensity  
- Confidence notes and limitations  

---

## Constraints

Current system limitations:

- **Single repetition per video**
- **Back squat only**
- **Single lateral camera**
- **Offline analysis only**

Some biomechanical variables cannot be inferred from monocular video, including:

- Knee valgus
- Foot pressure distribution
- Exact bar placement
- Lumbar stability

---

## Tech Stack

- Python  
- MediaPipe Pose  
- OpenCV  
- NumPy / SciPy  
