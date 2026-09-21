# Dataset Protocol

## 1. Design

The final dataset is a real-world paired-scene dataset:

- 10 unique scenes;
- 5 Development scenes;
- 5 Evaluation scenes;
- 3 acquisition conditions per scene: Light, Medium, Heavy;
- 30 real images total.

Development and Evaluation scenes are disjoint.

## 2. Capture principle

Scene geometry, object placement, background, camera position, and room illumination should be held approximately constant while acquisition settings, primarily ISO, are varied.

Do not claim that only noise changes. ISO is an acquisition-condition proxy.

Suggested target ranges:

```text
Light   ≈ ISO 100-400
Medium  ≈ ISO 400-1600
Heavy   > ISO 1600
```

These are practical acquisition targets rather than values that must be forced if the camera does not support them.

## 3. ROI protocol

Select a single homogeneous background ROI on the Light image.

The same coordinates must be reused for Medium and Heavy:

```text
same roi_x
same roi_y
same roi_w
same roi_h
```

ROI requirements:

- no strong object boundary;
- no text;
- no obvious pattern;
- low texture;
- sufficiently large to estimate a stable image-level residual.

The purpose is to reduce spatial/content confounding.

## 4. Noise proxy

For each paired image:

```python
blurred = cv2.GaussianBlur(roi, (5, 5), 0)
residual = roi.astype(np.float32) - blurred.astype(np.float32)
noise_measure = np.std(residual)
```

This is a **ROI-based high-frequency residual noise proxy**, not pure sensor noise. It may include contributions from compression, hidden camera processing, residual texture, and illumination changes.

## 5. Luminance

Compute mean luminance in the same ROI.

Relative difference from Light is used as an operational brightness check:

```text
abs(mean_level - mean_light) / mean_light * 100%
```

The project uses `<= 10%` as a team-defined operational criterion. This is not presented as a universal standard.

## 6. Non-monotonic proxy rule

The dataset must not be changed simply to force:

```text
Light < Medium < Heavy
```

for the noise proxy.

For example, a real observation such as:

```text
2.0 -> 3.1 -> 2.8
```

is retained and discussed. Possible explanations include smartphone computational photography, hidden denoising, sharpening, tone mapping, and compression.

Do not cherry-pick scenes.

Scenes may only be retaken/removed using criteria defined before reviewing the results, such as:

- camera movement;
- object movement;
- invalid capture settings;
- unreadable/corrupted file;
- invalid ROI;
- dimensions not compatible with the paired design.

## 7. Ground Truth

Ground Truth is manually annotated and independent of Canny.

Each scene has one GT file:

```text
scene01_gt.png
...
scene10_gt.png
```

All three paired images for a scene share the same `gt_id`.

## 8. Metadata schema

```text
image_id,split,noise_level,scene_id,gt_id,iso,roi_x,roi_y,roi_w,roi_h,noise_measure,mean_luminance,notes
```

All numeric measurements must be the actual measured values.

## 9. Strict validation

Before E1-E4, `validate_dataset.py` requires:

- exactly 30 active rows;
- exactly 10 scenes;
- exactly 5 Development and 5 Evaluation scenes;
- exactly Light/Medium/Heavy per scene;
- one GT per scene;
- identical ROI coordinates per scene;
- identical Light/Medium/Heavy image dimensions;
- valid files;
- valid numeric metadata;
- ROI inside image bounds.

Any critical failure terminates with exit code 1.
