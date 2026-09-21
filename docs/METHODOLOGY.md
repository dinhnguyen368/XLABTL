# Methodology

## 1. Challenge Statement

### Context

The task is edge detection on real-world images captured under different acquisition conditions that can produce varying levels of image noise.

### Challenge C

Real-world image noise associated with acquisition conditions.

### Failure F

Canny may generate spurious edges or miss weak edges when the input becomes noisy.

### Proposed technique P

ISO-based adaptive Bilateral Filtering followed by Canny.

### Mechanism

Bilateral filtering combines spatial proximity and intensity similarity. It is intended to smooth within similar-intensity regions while preserving important intensity transitions more effectively than indiscriminate smoothing.

### Predicted failure point

At sufficiently severe noise, the bilateral filter may no longer separate true structure from noise reliably, and Canny performance may degrade.

## 2. Baseline

```text
Grayscale -> Canny
```

Canny parameters are fixed by `config.yaml`.

## 3. Proposed method

```text
Grayscale
   -> ISO-based parameter selection
   -> Bilateral Filter
   -> Canny
```

The contribution is the context-specific adaptive parameter-selection rule, not a globally novel bilateral filtering algorithm.

## 4. ISO strata

The ISO thresholds are predefined **before Evaluation**:

```text
ISO <= 400           low_noise
400 < ISO <= 1600    medium_noise
ISO > 1600           high_noise
```

E2 tunes Bilateral parameters inside these predefined Development strata.

The ISO thresholds themselves are not optimized on Evaluation data.

## 5. Metrics

Primary metric:

**Strict F1**

Supplementary metrics may include:

- Precision;
- Recall;
- IoU;
- tolerance-based F1.

Visual inspection alone is not sufficient as evaluation.

## 6. Ground Truth independence

Ground Truth is manually annotated. Canny output is never used as Ground Truth.

## 7. Experiments

### E1

Demonstrate challenge evidence using paired Evaluation scenes and the Baseline only.

### E2

Perform parameter study on Development only. Freeze the resulting adaptive rule before Evaluation experiments.

### E3

Ablation comparison on Evaluation:

```text
Baseline = Grayscale -> Canny
Full     = Grayscale -> ISO-adaptive Bilateral -> Canny
```

### E4

Controlled Gaussian stress test starting from Light Evaluation images converted to grayscale. The Proposed method uses the frozen Heavy configuration.

## 8. Integrity rules

Experimental results must not be used to retroactively change dataset selection criteria, Ground Truth, metrics, baseline, proposed method, or E1-E4 protocol.

Unexpected results are reported rather than removed solely because they do not support the hypothesis.
