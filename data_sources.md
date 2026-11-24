# Data sources: paper-04 — checkpoint 3

Written 2026-11-22 by Taha. "Verified" means read that day from the dataset's own page, its readme, or the downloaded file itself. Evidence: `decisions/evidence/005_file_inventory.out.txt` (variables and shapes) and `006_group_counts.out.txt` (group counts and reference-value ranges). No spectrum has been plotted or related to a reference value; no model has been fitted.

## 1. Copenhagen tablet set (scale shift, batch-held-out testing)
- **Name:** "Active substance in pharmaceutical tablets", NIR transmittance and Raman spectra (development data).
- **Owner:** Chemometrics Research group, University of Copenhagen; data from Dyrby, Engelsen, Nørgaard, Bruhn and Lundsberg Nielsen.
- **URL:** https://ucphchemometrics.com/tablet/ → `Tablets.zip` on the university's file share (verified). Hash in section 3.
- **Licence:** `[CHECK]` none stated on the page or in the readme (verified). Offered publicly with a reference to cite. Our use is research; the data is not redistributed: the code repository will hold a download script and the hash.
- **Access level:** open, no account (verified).
- **Citation required:** Dyrby M, et al. Applied Spectroscopy 2002;56(5):579-585 (from the page; DOI not yet looked up; older than 8 years, allowed as the original source of the dataset).
- **Content (verified in the file `NIRdata_tablets.MAT`):**
  - 310 tablets, 404 spectral variables from 7398.3 to 10507.3 cm⁻¹, no missing values.
  - Per tablet: active substance content in % w/w (range 4.61 to 9.79), `Type` (codes 1-4: 70, 80, 80, 80 tablets), `Scale` (codes 0, 1, 2: 70, 120, 120 tablets). The readme names the levels: scale "laboratory, pilot, full", type "A, B, C, D".
  - Tablet labels have the form `GG_TT`. **There are 31 groups `GG` of exactly 10 tablets, and every group has a single type and a single scale** (7 groups at scale code 0, 12 at code 1, 12 at code 2; three groups per type at codes 1 and 2).
  - The Raman file holds 120 of the same tablets (12 groups); it is not used.
- **Not verified:**
  - `[CHECK]` **That a group `GG` is a production batch.** It is the reading the structure suggests (10 tablets, one type, one scale); neither the page nor the readme says so. To be confirmed in Dyrby 2002 before the paper uses the word "batch". Until then the plan says "tablet group".
  - `[CHECK]` **Which scale code is which level.** The readme's order suggests 0 = laboratory, 1 = pilot, 2 = full. Within-group spread of the content is widest at code 0 and narrowest at code 2, which fits that order, but this is an inference. To be confirmed in Dyrby 2002.
  - What tablet types A-D are (the content suggests type 1 is a different strength: about 5% w/w against about 8%).
  - How the reference content was measured, and its error.
- **Known issues:** the content range at scale code 2 reaches 9.79% w/w, above the largest value at codes 0 and 1 (8.58%), so a model built on the other scales must extrapolate for some tablets. Ali 2023 (doi:10.1208/s12249-022-02493-5) has used this set for laboratory-to-production testing.
- **What we use:** the NIR spectra, the content, type, scale and the group label.

## 2. IDRC 2002 "shootout" tablets (instrument shift)
- **Name:** NIR spectra of pharmaceutical tablets from the 2002 International Diffuse Reflectance Conference "Shootout".
- **Owner:** published by the IDRC; this copy converted and hosted by Eigenvector Research, Inc.
- **URL:** https://eigenvector.com/resources/data-sets/ → `nir_shootout_2002.mat_.zip` (verified). Hash in section 3. The original IDRC page (idrc-chambersburg.org/shootout_2002.htm) answers "403 forbidden" to a script; `[CHECK]` the human may open it in a browser to read the original description.
- **Licence:** `[CHECK]` none stated (verified). The host describes it as "a great data set for chemometrics training and algorithm testing". Same handling as above: cited, not redistributed.
- **Access level:** open, no account (verified).
- **Content (verified in the file):**
  - Three parts as split by the organisers: calibration 155 tablets, validation 40, test 460: **655 tablets** (the host's page says 654).
  - Each tablet measured on two spectrometers (`_1`, `_2`): 650 variables, 600 to 1898 nm in steps of 2 nm; the two instruments' spectra differ.
  - Three reference values per tablet: weight, hardness, assay. Assay range: calibration 151.6 to 239.1, validation 188.2 to 199.5, test 154.3 to 237.7.
  - **No batch, class or sample label of any kind** (all class and label fields are empty).
- **Not verified:** the units of the three reference values; the active ingredient; how the organisers chose the three parts; which tablets share a production batch; the instrument makes.
- **Known issues:** the validation part covers a narrow assay range (sd 2.4 against 22.0 in calibration), so it is a poor tuning set on its own. Because batches are unknown, tablets cannot be assumed independent; the paper says so.
- **What we use:** both instruments' spectra and the assay value. Weight and hardness are not used.

## 3. Snapshot (downloaded 2026-11-22 12:01 UTC, scripted curl, approved by the human that day)
| file in `data/raw/` | URL | bytes | sha256 |
|---|---|---|---|
| `idrc2002_shootout/nir_shootout_2002.mat_.zip` | https://eigenvector.com/wp-content/uploads/2019/06/nir_shootout_2002.mat_.zip | 5,414,232 | `04c03a4314ab8e8f63e0ecc788b657b5d1b5450791ce465ad41ad90cb5d52125` |
| `ucph_tablets/Tablets.zip` | https://sid.erda.dk/share_redirect/HbpPvIEvTg/Tablets.zip | 3,822,359 | `b583e3ad176ad6571d2f38c63dc4af3e9a9f283d59d004f85f3ae85d7a212289` |

Both are static files (archive entries dated 2019 and 2022; data dated 2004), not versioned or appended sources. Extracted by unzip only, next to each archive. `data/raw/` is read-only and not in git; a manifest file could not be written there (the folder is write-protected for agents), so this table is the manifest. Notebook 01 re-checks both hashes and writes the hashes of the extracted files to `outputs/`.

## Consequence for the title
The council's condition was that the title match the labels the data really contain. "Batch-held-out" is possible on the Copenhagen set only, and only once Dyrby 2002 confirms that the 31 groups are batches. The shootout set supports "instrument-held-out", nothing about batches.

## What the human is asked at checkpoint 3
1. Approve these two sources as the paper's data.
2. If you have access to Applied Spectroscopy, open Dyrby 2002 (56(5):579-585) so that the two `[CHECK]` points (groups = batches; scale codes) can be confirmed from the paper. Without it the paper says "tablet group" and names the scales by the readme's order, marked as the dataset documentation's order.
