# CONTEXT.md - Domain Model & Architectural Glossary

## 1. Domain Entities & Clinical Vocabulary

- **Cervix Image**: Raw colposcopic visual inspection with acetic acid (VIA/IVA) digital image captured under LED illumination. Subject to glares, specular reflections, vaginal wall obscuration, and bodily fluids.
- **Squamo-Columnar Junction (JSC / SCJ)**: Critical anatomical transformation zone of the uterine cervix where cervical intraepithelial neoplasia (CIN) develops.
- **Type 1 / Type 2 / Type 3 (WHO Cervical Types)**:
  - *Type 1 (Normal / Low Risk)*: SCJ is completely visible and ectocervical.
  - *Type 2 (Lesional / Intermediate Risk)*: SCJ is fully visible with endocervical component.
  - *Type 3 (Incomplete / Non-Visualizable)*: SCJ is not fully visible, requiring clinical referral or colposcopy.
- **SaMD (Software as a Medical Device)**: Clinical decision support classification pipeline subject to strict safety constraints: Sensitivity (Recall) >= 95%, Specificity >= 80%, F2 >= 0.88.
- **Clinical Triage Engine**: Tri-class decision rule mapping calibrated risk probabilities to action categories:
  - *Green ($P < 0.20$)*: Negative (Routine 3-year follow-up).
  - *Yellow ($0.20 \le P < T_{\text{optimal}}$)*: Equivocal / Second wash or second opinion needed.
  - *Red ($P \ge T_{\text{optimal}}$)*: High risk / Immediate referral or treatment.

---

## 2. Deep Modules & Architecture Vocabulary

- **Cervical Image Pipeline (`CervixImagePipeline`)**:
  - *Depth*: Deep module encapsulating specular reflection suppression, procedural Perlin noise artifact injection (blood/mucus), strict WHO-compliant color jitter ($\text{Hue} \le 0.05$), and normalization into standardized PyTorch tensors `[3, 224, 224]`.
  - *Interface*: `pipeline.process(image_or_path, is_train=True) -> torch.Tensor`.
- **IVA Dataset (`IVADataset`)**:
  - *Responsibility*: PyTorch `Dataset` indexation and sample retrieval. Delegates all image transformations to `CervicalImagePipeline`.
- **CADe Detector (`JSCDetectorStage1`)**:
  - *Responsibility*: Stage 1 object localization of the Squamo-Columnar Junction with 15% safety padding and 70% center crop fallback.
- **CADx Classifier (`IVALesionClassifierStage2`)**:
  - *Responsibility*: Stage 2 ConvNeXt vision backbone producing unnormalized 3-class logits.
- **Clinical Triage Evaluator (`ClinicalTriageEvaluator`)**:
  - *Responsibility*: Pareto-optimal threshold calibration ($T_{\text{optimal}}$ for Sensitivity $\ge 95\%$) and clinical triage compliance reporting.
