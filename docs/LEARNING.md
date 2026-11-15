# Learning notes

One section per core item from the brief ("you implement yourself"). Each has the maths, the design choice, the alternatives rejected, and three interview questions with short answers. Read a section, close it, and explain the item out loud. If you can't, read the code it points to.

---

## 1. Preprocessing filters

Code: `src/ecgq/filters.py`, tests: `tests/test_filters.py`, figure: `reports/figures/filter_responses.png`.

**What and why**
- **Baseline wander** (breathing, movement, electrode drift) sits below about 0.5 Hz. A Butterworth high-pass at 0.5 Hz removes it. The cut-off is a compromise: the ST segment and T wave carry energy down to the heart rate (about 0.67 Hz at 40 bpm), so going higher distorts the ST segment.
- **Mains interference** is a narrow line at 50 Hz (60 Hz in the USA). An IIR notch has transfer function H(z) = (1 - 2cos(w0) z^-1 + z^-2) / (1 - 2r cos(w0) z^-1 + r^2 z^-2): zeros on the unit circle at the mains frequency, poles just inside at the same angle so only a narrow band is cut. Q = f0 / bandwidth = 30 gives about 1.7 Hz bandwidth.
- A **40 Hz low-pass** removes muscle noise above the useful ECG band before resampling and feature extraction.
- **Zero phase**: `sosfiltfilt` runs the filter forward then backward. The phase responses cancel, so peaks do not shift in time, and the magnitude response is squared (effective order doubles). This only works offline, because it needs future samples.
- **Second-order sections (SOS)** are used instead of a single (b, a) polynomial: high-order IIR filters in direct form are numerically fragile, especially with low cut-offs relative to fs.
- **Resampling** uses a polyphase filter (`resample_poly`), which applies an anti-aliasing low-pass before decimation. Plain slicing (`x[::4]`) would alias muscle noise above 125 Hz into the ECG band.

**Alternatives rejected**
- Median-filter baseline removal (two cascaded medians of 200 ms and 600 ms): good at keeping ST shape but slow on 24 h records and harder to analyse in frequency.
- Adaptive notch or LMS cancellation of mains: useful when the mains frequency drifts, unnecessary here.
- Wavelet denoising: the project's goal is to judge quality, not to repair the signal.

**Interview questions**
1. *Why 0.5 Hz and not 1 Hz for the high-pass?* Because ST-segment and T-wave energy reaches down to about 0.67 Hz; a higher cut-off distorts the ST level, which matters clinically. 0.5 Hz with a zero-phase filter is a common compromise for monitoring; diagnostic 12-lead ECG uses a much lower cut-off (around 0.05 Hz) to protect the ST segment completely.
2. *What does filtfilt do to the filter order and phase?* It squares the magnitude response (doubling the effective order and the attenuation in dB) and gives exactly zero phase, so no delay. The cost is non-causality.
3. *Why not slice to downsample from 1000 Hz to 250 Hz?* Slicing keeps every 4th sample without removing content above the new Nyquist (125 Hz), so that energy aliases into the band you analyse. A polyphase resampler low-passes first.

---

## 2. Pan-Tompkins QRS detector

Code: `src/ecgq/detectors/pan_tompkins.py`, figure: `reports/figures/pan_tompkins_stages.png`, results: `reports/metrics/qrs_mitdb_summary.json`.

**The stages (200 Hz, as in the paper)**
1. **Low-pass** H(z) = (1 - z^-6)^2 / (1 - z^-1)^2: two 6-point running sums in cascade, a triangle with 11 taps. Cut-off about 11 Hz, delay 5 samples, gain 36.
2. **High-pass** H(z) = z^-16 - (1/32)(1 - z^-32)/(1 - z^-1): a 16-sample delay minus a 32-point moving average. Cut-off about 5 Hz, delay 16 samples. Together: a band-pass of roughly 5 to 15 Hz, where QRS energy is concentrated and P/T waves and baseline are weak.
3. **Derivative** y(n) = [2x(n) + x(n-1) - x(n-3) - 2x(n-4)] / 8: approximates d/dt, emphasising the steep QRS slopes.
4. **Squaring** makes every value positive and amplifies large slopes non-linearly, so QRS dominates T waves.
5. **Moving-window integration** over 30 samples (150 ms, about the widest normal QRS) turns each QRS into one smooth hump.
6. **Decision**: candidate peaks of the integrated signal at least 200 ms apart (refractory period). Two running estimates per signal: SPKI = 0.125 PEAKI + 0.875 SPKI after a QRS, NPKI = 0.125 PEAKI + 0.875 NPKI after noise. Threshold I1 = NPKI + 0.25 (SPKI - NPKI). The same is done on the band-passed signal (F thresholds); a QRS must clear both.
7. **Search-back**: keep the average of the last 8 RR intervals that were within 92 to 116 % of the average (RR AVERAGE2). If nothing is found for 166 % of that average, the largest candidate since the last beat that clears the lower threshold I2 = 0.5 I1 becomes a beat, and SPKI is updated with weight 0.25.
8. **T-wave rejection**: a candidate within 360 ms of the last beat with a maximum slope under half the last QRS's slope is a T wave.
9. **Irregular rhythm**: if the overall RR average falls outside the 92 to 116 % band of the regular average, thresholds are halved.

**Design choices in this repo**
- The paper's filters are recursive with poles on the unit circle cancelled by zeros. Over 17 million samples, float round-off would accumulate. The same transfer functions are applied as FIR filters; a unit test proves the outputs match.
- Detections are mapped back through the filter delays (21 samples), rescaled to the input rate and refined to the largest absolute deflection of a 5 to 25 Hz band-pass within 60 ms. Timing error on MIT-BIH is 2.7 ms on average (EC57 protocol).
- **Bug found and fixed** (see `docs/DECISIONS.md`, D2): a false beat during filter warm-up made the first RR interval short, so search-back fired after every normal beat and picked up T waves, because only normal detections got the T-wave test. Record 117 had 1,460 false positives; after the fix, zero. This is a good interview story: how you found it (per-record table sorted by errors), why it happened, and the minimal fix.

**Alternatives rejected**
- Using a library implementation (NeuroKit2 `pantompkins1985`): the brief requires writing it from the paper.
- Zero-phase band-pass instead of the paper's integer filters: easier, but no longer the published algorithm, and the T-wave and threshold logic is tuned to that filter's output.

**Interview questions**
1. *Why square instead of taking the absolute value?* Squaring is non-linear: it boosts large derivative values (QRS) far more than small ones (P and T waves, noise), so the integrated QRS hump stands well above everything else. Absolute value keeps the ratio linear.
2. *What does the 150 ms integration window trade off?* Too short and one QRS gives several peaks; too long and the QRS merges with the T wave and the timing smears. 150 ms is about the widest normal QRS at 200 Hz (30 samples).
3. *Why two threshold sets (integrated and band-passed)?* The integrated signal is robust but blurred in time; the band-passed signal keeps the QRS shape. Requiring both cuts false positives from wide, slow artefacts that integrate to a large value but have no sharp band-passed peak.

---

## 3. Elgendi two-moving-average detector

Code: `src/ecgq/detectors/elgendi.py`.

**Maths**: band-pass 8 to 20 Hz (third-order Butterworth, zero phase), square, then MA_qrs (W1 = 97 ms) and MA_beat (W2 = 611 ms). Blocks of interest: MA_qrs > MA_beat + beta * mean(y^2), beta = 0.08. Keep blocks at least W1 wide; the R peak is the largest |filtered| value inside each block. A 300 ms refractory rule removes double detections.

**Why it is here**: it is a second, independent detector with different failure modes (no adaptive memory, no search-back), which makes their agreement (bSQI) meaningful. It is also simple and fast: 4 s for all 48 MIT-BIH records.

**Watch-out**: the offset uses the mean energy of the whole input. On a 24-hour recording, one long artefact raises that mean for the whole day. On BUT QDB both detectors therefore run in 60 s blocks (D5).

**Interview questions**
1. *Why two moving averages?* The short one follows the QRS energy, the long one estimates the local energy around a whole beat. A QRS is where short-term energy rises above the local average; the comparison is self-normalising to amplitude.
2. *Why reject blocks narrower than W1?* A real QRS keeps its energy high for about one QRS width; narrow blocks are spikes or noise.
3. *Elgendi scores higher than Pan-Tompkins on MIT-BIH here. Why keep Pan-Tompkins?* It is the reference algorithm, it has the best timing accuracy here (2.7 ms vs 4.9 ms), and the project's quality index needs two detectors that fail differently.

---

## 4. Detector-agreement quality index (bSQI) and the other SQIs

Code: `src/ecgq/metrics.py` (`detector_agreement`), `src/ecgq/quality/features.py`.

**bSQI** (Li, Mark and Clifford 2008): match beats from two detectors one-to-one within 150 ms. bSQI = N_matched / (N_1 + N_2 - N_matched), the intersection over the union of the two beat sets. Clean ECG: both detectors find the same beats, bSQI near 1. Noise: each detector is fooled differently, bSQI drops.

**Other indices (17 in total, all scale-invariant)**
- kSQI: kurtosis. Clean ECG is peaky (sharp QRS, flat baseline), kurtosis well above 5. Gaussian noise has kurtosis 3.
- sSQI: absolute skewness (QRS spikes make the distribution asymmetric).
- pSQI = P(5 to 15 Hz) / P(5 to 40 Hz): QRS energy share.
- basSQI = 1 - P(0 to 1 Hz) / P(0 to 40 Hz) on the signal before the high-pass: how little baseline wander there is.
- tSQI: mean correlation of each beat (+-250 ms) with the median beat of the window (Orphanidou 2015). Measures morphological consistency.
- Heart rate from each detector, their difference, RR coefficient of variation, longest gap without a beat: physiological plausibility.
- High-frequency share above 40 Hz (muscle noise), spectral entropy, flat-line fraction, zero-crossing rate, ratio of the 99.5th percentile amplitude to the median R amplitude, QRS-to-background energy ratio.

**Why scale-invariant**: amplitude depends on the device, the units (microvolts vs millivolts), electrode contact and the person. Ratios transfer between datasets.

**Interview questions**
1. *Why is detector agreement a quality measure at all?* Two detectors with different designs agree on clean ECG; noise fools them in different ways, so disagreement is evidence of noise without knowing the true beats.
2. *Can bSQI be high on bad signal?* Yes: if both detectors fail the same way (for example both lock onto a periodic artefact, or both find nothing). That is why it is combined with template correlation, heart-rate plausibility and spectral indices.
3. *Why is kurtosis useful?* ECG is mostly flat with rare sharp peaks, a heavy-tailed distribution. Noise and motion artefact make the distribution closer to Gaussian, so kurtosis falls.

---

## 5. Accelerometer features

Code: `acc_features` in `src/ecgq/quality/features.py`.

Seven features per 10 s window: standard deviation of the acceleration magnitude, mean absolute jerk (derivative of the magnitude), largest per-axis standard deviation, log power 0.5 to 3 Hz (walking cadence) and 3 to 15 Hz (impacts, tremor), 1st to 99th percentile range, and tilt change (angle between the mean gravity vector in the first and last second, a posture change).

**Interview questions**
1. *Why the magnitude rather than single axes?* The patch orientation differs between people and changes with posture; the magnitude does not depend on orientation.
2. *Can motion exist without ECG artefact, and the reverse?* Yes to both. A well-attached patch during walking can give clean ECG; a loose electrode can give artefact while the person is still. That is exactly why accelerometer features may help less than expected.
3. *Why could the accelerometer still help for unseen subjects?* ECG artefact looks different in each person, while motion physics is similar for everyone, so motion features might generalise better than ECG features. The LOSO test checks this.

---

## 6. Classifiers

Code: `src/ecgq/quality/models.py`, results: `reports/metrics/quality_loso_summary.json`.

**Rule-based**: class 3 if bSQI < b3 or heart rate outside 30 to 220 bpm; class 1 if bSQI >= b1 and tSQI >= t1 and basSQI >= s1; class 2 otherwise. The four thresholds are fitted by grid search (macro-F1, weighted) on the training subjects of each fold. Fully interpretable; the baseline every other model must beat.

**Gradient boosting** (`HistGradientBoostingClassifier`): an additive model of shallow trees, F_m(x) = F_{m-1}(x) + eta * h_m(x), where each tree h_m fits the negative gradient of the multinomial log-loss. Learning rate 0.05, 300 trees, at most 31 leaves, at least 40 samples per leaf, L2 regularisation 1. Features are binned into histograms (fast). It handles non-linear interactions between SQIs that fixed thresholds cannot.

**1D CNN**: four blocks of (convolution, batch norm, ReLU, max-pool by 2) with 16, 32, 64, 64 channels and kernels 7, 7, 5, 5. Then global average and global max pooling (concatenated, 128 values), dropout 0.3, and a linear layer to 3 classes. About 35k parameters. Input: 10 s at 100 Hz, ECG scaled by its median and interquartile range, optional three accelerometer channels. Trained with AdamW (lr 1e-3, weight decay 1e-3), cosine learning-rate decay, 6 epochs of 8,000 windows drawn with class-and-record-balanced probabilities. Augmentation: polarity flip, amplitude scaling 0.8 to 1.2, small Gaussian noise.

**Class imbalance and long records** (D8): sample weight = 1/(class count) x 1/sqrt(record count), normalised.

**Interview questions**
1. *Why macro-F1 and not accuracy?* Class 1 dominates; a model that always says class 1 has high accuracy and is useless. Macro-F1 averages the F1 of each class, so class 3 counts as much as class 1.
2. *Why global max pooling as well as average pooling?* An artefact can occupy one second of the ten; averaging would dilute it, max pooling keeps the strongest local evidence anywhere in the window.
3. *Gradient boosting vs random forest?* A forest averages independent deep trees (reduces variance); boosting adds shallow trees sequentially, each correcting the previous errors (reduces bias). Boosting usually wins on tabular features like SQIs, at the cost of more tuning.

---

## 7. Subject-wise splits and the statistics of the research question

Code: `scripts/evaluate_quality.py`.

**Leave-one-subject-out (LOSO)**: 15 folds; in each, every record of one subject is the test set and the other 14 subjects train the model (and fit the rule thresholds). Windows from the same person are correlated (same anatomy, electrode placement, habits), so a random window split puts near-duplicates on both sides and overstates performance. LOSO answers the real question: how well does it work on a new person?

**Paired comparison**: for the question "do accelerometer features help on unseen subjects?", each subject gives one macro-F1 with and one without accelerometer features, so the comparison is paired. The Wilcoxon signed-rank test ranks the absolute per-subject differences and checks whether positive and negative ranks balance; it does not assume normality, which suits 15 values. A bootstrap over subjects (resample the 15 subjects with replacement 2,000 times, recompute the pooled macro-F1 difference) gives a 95 % confidence interval that respects the clustering of windows within subjects.

**Interview questions**
1. *Why not split windows at random?* Adjacent windows of the same person are nearly identical; the model can memorise the person and the score says nothing about new users. Several published BUT QDB results have this problem.
2. *Why bootstrap subjects, not windows?* Windows within a subject are not independent. Resampling windows would treat 30,000 correlated windows as independent evidence and give a falsely narrow interval; the real sample size is 15 subjects.
3. *What did this project conclude about the accelerometer?* With gradient boosting: a small pooled gain, CI [+0.002, +0.037], but only 9 of 15 subjects improved (p = 0.23). With the CNN: worse for 10 of 15 subjects (p = 0.048). So no consistent benefit on unseen people; the likely reason is that most class 3 ECG here happens while the person is still.
4. *What would you conclude from p = 0.3 with a confidence interval spanning zero?* No evidence of an effect at this sample size, which is not evidence of no effect. Report the interval: it shows how large an effect the data can still rule out.

---

## 8. Evaluating QRS detection (EC57 conventions)

- One-to-one matching: each reference beat claims the nearest unused detection within 150 ms. Unmatched references are false negatives, unmatched detections false positives.
- Se = TP / (TP + FN) (how many real beats were found), +P = TP / (TP + FP) (how many detections were real).
- Ventricular flutter episodes are excluded (no beat labels inside them). The EC57 protocol skips the first 5 minutes as the learning period. Paced records are reported with and without.
- Timing error of matched beats is reported too, because a 150 ms window hides timing problems (Porr and Macfarlane 2024).

**Interview question**: *Why is the gross (pooled) Se different from the average of per-record Se?* Pooling weights each record by its number of beats; averaging per record weights every record equally, so a short, difficult record moves the average more. EC57 asks for both.

---

## 9. The quality gate, and three things that went wrong

Code: `scripts/evaluate_rejection.py`, figure: `reports/figures/rejection_nstdb.png`, numbers: `reports/results_tables.md`.

**The gate**: score every 10 s window, drop class 3 windows, run beat detection only on the rest. On the noise stress test database (MIT-BIH 118 and 119 with electrode-motion noise), Pan-Tompkins F1 at 0 dB rises from 0.814 to 0.910 with the GBM gate, keeping 72 % of beats, and no clean window is rejected. The trade-off is coverage: you lose the beats in rejected windows. An "oracle" that knows exactly where noise was added shows the ceiling.

**What went wrong, and what each one teaches**
1. *Pan-Tompkins search-back bug* (record 117, D2): one wrong early beat poisoned the RR average. Lesson: adaptive algorithms need their start-up handled; sort results per record to find failures.
2. *Mislabelled-looking class 3* (record 105001, D14): 13 hours of a detached device made class 3 look easy (F1 0.96). After using the authors' benchmark ranges, class 3 F1 is 0.63. Lesson: plot the whole recording before trusting label counts; follow the dataset authors' protocol.
3. *CNN gate rejects an arrhythmia* (record 118, D17): a CNN trained on BUT QDB has never seen right bundle branch block, so it calls a perfectly clean RBBB recording unusable. Lesson: "unusual" and "noisy" look the same to a model that only saw normal beats. This is the same confound the AliveCor manual warns about (flutter and bigeminy reported as unreadable).

**Interview questions**
1. *Why might the simple rule gate transfer to MIT-BIH as well as gradient boosting?* It uses three thresholds on detector agreement, template correlation and baseline power, which mean the same thing on any ECG. The boosted model also uses finer, device-specific feature interactions learned on BUT QDB, which shift on a Holter recording from another lead and population.
2. *How would you choose the rejection threshold for a product?* From the cost of each error: a false alarm (bad beat accepted) versus missing data (good beat rejected). Plot F1 or +P against coverage for different probability thresholds and pick the operating point the application can tolerate, then validate it on new subjects.
3. *Why is a 95 % confidence interval that excludes zero not enough to claim the accelerometer helps?* The bootstrap interval is for the pooled score, which large subjects dominate; the per-subject test (9 better, 6 worse, p = 0.23) says the gain is not consistent across people. Both views are reported; neither alone is the whole story.
