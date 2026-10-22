# PhyVision Model Card: PhyNet v5.1

**Author:** Swayam Shakya · **License:** MIT · **Weights:** `models/phyvision_phynet.pt` (PyTorch)

## What it does

Classifies the physique of one standing person in a photo into **Athletic, Normal, Obese or Skinny**, from the person's silhouette, and estimates how much of the body the photo shows (the framing).

## Architecture

| Part | Details |
|---|---|
| Silhouette branch | torchvision ResNet-18, 1-channel 7×7 stem, trained from scratch, global-average-pooled to 512 features |
| Profile branch | 100-d shape profile (total and central-run width at 48 heights, plus aspect, fill, relative area, hull solidity) → Linear 128 → BN → ReLU → Dropout 0.2 → Linear 64 → ReLU |
| Fusion | concat 576 → BatchNorm → Dropout 0.4 |
| Class head | Linear 128 → ReLU → Dropout 0.2 → Linear 4 |
| Framing head | Linear 64 → ReLU → Linear 1 → sigmoid: fraction of body height visible (1.0 = head to feet) |
| Test-time | average of the silhouette and its mirror image |
| Parameters | 11.3 million |

## Training

**Body classes.** The class body proportions were recalibrated in v5.1 against adult anthropometric ratios after an earlier v5.1 model, trained on caricatured classes, called a real lean person "Normal". For a 1.75 m man, upper-arm circumference is about 24 cm (Skinny), 31 cm (Normal), 40 cm (Athletic) and 42 cm (Obese). Realistic classes are closer together, so test accuracy is lower than v5.0's, but the model transfers better to real photos.

**Stage 1, shape.** 8,000 exact training masks covering four framings and many poses, corrupted on the fly to imitate segmentation errors (flip, erode or dilate, ragged edges, stray blobs, holes, extra bottom crop, rotation, slight horizontal scaling). Loss = cross-entropy with label smoothing 0.05 + 2 × L1 framing error. AdamW (lr 2e-3, wd 1e-4), one-cycle cosine schedule, batch 64, 16 epochs. Best validation accuracy 95.47%.

**Stage 2, real segmenters.** Every training photo was cut out again from scratch, half by rembg (`u2net_human_seg`) and half by two-pass GrabCut (7,985 masks). PhyNet was fine-tuned on a 50/50 mix of exact and segmenter masks in two rounds: 6 epochs at lr 4e-4 using the first half of the cut-outs, then 6 epochs at lr 3e-4 using all of them (12 epochs in total). Validation accuracy on segmenter masks went from 72.7% to 87.4%, with 96.1% on exact masks. Within each round the checkpoint was chosen on the mean of the two (validation people only). See `scripts/build_segmenter_masks.py` and `src/phyvision/finetune.py`.

**Weight averaging.** The released weights are the average of the best round-1 checkpoint (fine-tuning epoch 5) and the best round-2 checkpoint (epoch 10), made with `scripts/average_checkpoints.py`. With mirror averaging, on validation:

| Model | Segmenter masks (799) | Exact masks (4,000) |
|---|---:|---:|
| Stage 1 only | 74.1% | 96.1% |
| Round 1 best | 87.5% | 96.2% |
| Round 2 best | 88.2% | 97.1% |
| Average of rounds 1 and 2 (released) | 88.2% | 96.8% |

The averaged model and round 2 alone are tied on validation. The tie was broken with the one real photo described below: round 2 alone flipped between Skinny and Athletic near 50% across variants of that photo, while the averaged model returned Skinny for all 12 variants. Because one photo influenced this choice, it is not an independent test.

The test split was used once, after training, for the numbers below.

## Results (held-out test split)

| Evaluation | Test images | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Exact masks (classifier only) | 4,000 | **97.00%** | 0.970 |
| End to end, rembg segmentation | 600 | **88.00%** | 0.880 |
| End to end, GrabCut segmentation (offline) | 600 | **81.33%** | 0.813 |

Per class:

| Class | Precision | Recall | F1 | F1 with rembg | F1 with GrabCut |
|---|---:|---:|---:|---:|---:|
| Athletic | 0.983 | 0.987 | 0.985 | 0.885 | 0.822 |
| Normal | 0.936 | 0.946 | 0.941 | 0.825 | 0.724 |
| Obese | 1.000 | 0.998 | 0.999 | 0.944 | 0.913 |
| Skinny | 0.961 | 0.949 | 0.955 | 0.868 | 0.794 |

Accuracy by framing:

| Framing | Exact masks | End to end, rembg | End to end, GrabCut |
|---|---:|---:|---:|
| Head to feet | 97.7% (2042) | 88.2% (305) | 80.0% (305) |
| Head to knees | 97.4% (575) | 95.2% (83) | 81.9% (83) |
| Head to mid-thigh | 95.2% (774) | 86.7% (113) | 79.6% (113) |
| Waist up | 96.4% (609) | 82.8% (99) | 86.9% (99) |

Framing estimate on the test set: mean absolute error 0.004 of body height.

* **Exact masks** isolate the classifier: all 4,000 test people with their reference masks.
* **rembg / GrabCut** are end to end: 600 test photos (150 per class) segmented from scratch, as an upload would be.

## Real-photo spot check

Four real stock photos, one per class, were tested during development: a lean young man flexing one arm, a heavy-set man with hands on hips, a muscular shirtless man against a dark background, and an average-build man with hands in his pockets. All are small (about 180 × 250 px), cropped between the knees and mid-thigh, and carry stock-site watermarks. They are licensed previews, so they are not included in the repository.

| Photo | Expected | rembg (default) | GrabCut (offline) | Framing detected |
|---|---|---|---|---|
| Lean man flexing one arm | Skinny | **Skinny 54.5%** | **Skinny 66.9%** | Head to mid-thigh |
| Heavy-set man, hands on hips | Obese | **Obese 87.6%** | **Obese 87.8%** | Head to knees |
| Muscular man, shirtless, dark background | Athletic | **Athletic 83.8%** | **Athletic 56.2%** | Head to mid-thigh |
| Average build, hands in pockets | Normal | **Normal 85.5%** | **Normal 90.4%** | Head to mid-thigh |

Each photo was also tested in 7 variants (original, mirrored, upscaled 3×, cropped tighter, extra margin, greyscale, JPEG quality 50) with both segmenters: 53 of 56 correct overall (skinny 14/14, obese 14/14, athletic 12/14, normal 13/14). The remaining misses are segmentation failures rather than classifier errors: on the dark-background athletic photo, rembg sometimes fills the gap between arm and waist, which erases the V-taper; on the normal photo, GrabCut once dropped the jeans.

Four photos are an anecdote, not an accuracy measurement. The skinny photo was also used to choose between two validation-tied models (see Weight averaging), and the variant test led to one inference change (enlarging small photos before segmentation). A third fine-tuning round with an extra "filled-gap" segmentation augmentation was tried and rejected: it tied on validation and did worse on these photos.

## Intended use

* Personal fitness education and exploring computer vision.
* Coursework and research on body-shape classification and synthetic data.

## Out of scope

* Medical diagnosis, body-fat or BMI estimation, health or insurance decisions.
* Judging, ranking or profiling people, or analyzing anyone without their consent.
* Children.

## Limitations

* **Domain gap.** Trained only on rendered people. Real photos differ in texture, clothing and pose. Accuracy on real photographs has not been measured on a labelled set and will be lower than above; it depends mostly on following the photo guide and on how cleanly the person is segmented. rembg is strongly recommended over GrabCut.
* **2-D outline only.** Muscle and fat with the same outline look the same. Loose clothing, side or back views, sitting, waist-up crops and groups reduce reliability. The photo check flags these.
* **Coarse labels.** Real bodies are a continuum. Results under 60% confidence are flagged as borderline.
* **Calibration.** Label smoothing caps confidence near 96%. Treat confidence as relative, not as a calibrated probability, especially on real photos.

## Fairness notes

The classifier never sees colour, so skin tone, hair and clothing colour cannot influence it directly. The generator covers 12 skin tones, both sexes and varied clothing, and the frame size is jittered per person. Body-shape norms differ across populations and ages, and the class boundaries are a design choice rather than a medical standard.

## Dataset and training provenance

The training images are computer-generated by PhyVision's procedural renderer (see `docs/DATASET_CARD.md`). The dataset generator and the training pipeline were developed with help from Claude, an AI model by Anthropic, as a learning aid for computer vision.
