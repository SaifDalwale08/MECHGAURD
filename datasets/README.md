# MECHGUARD Datasets

Raw dataset files are **not stored in this repository** due to their large size.

Download the datasets from the official sources below and place them in the
corresponding subdirectories before running the training scripts.

---

## CWRU Bearing Dataset

- **Location:** `datasets/CWRU/`
- **Source:** [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file)
- **Files used:** `97.mat`, `98.mat`, `105.mat`, `118.mat`, `130.mat`

---

## Paderborn Bearing Dataset

- **Location:** `datasets/Paderborn/`
- **Source:** [Paderborn University Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download)
- **Bearing types used:** K001 (healthy), KA01 (outer race damage), KB23 (inner race damage), KI01 (rolling element damage)

---

## NASA Milling Dataset

- **Location:** `datasets/NASA_Milling/3.+Milling/3. Milling/`
- **Source:** [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- **File:** `mill.zip`

---

## After Downloading

Ensure the directory structure matches what the training scripts expect:

```
datasets/
├── CWRU/
│   ├── 97.mat
│   ├── 98.mat
│   ├── 105.mat
│   ├── 118.mat
│   └── 130.mat
├── Paderborn/
│   ├── K001/K001/*.mat
│   ├── KA01/KA01/*.mat
│   ├── KB23/KB23/*.mat
│   └── KI01/KI01/*.mat
└── NASA_Milling/
    └── 3.+Milling/3. Milling/
        └── mill.zip
```
