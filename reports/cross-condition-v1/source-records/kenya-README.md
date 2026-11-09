# Kenya Forest Sounds Dataset

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## Authors and Contributors

All authors and contributors are listed on Zenodo:

- https://zenodo.org/records/21443074

---

## About

The **Kenya Forest Sounds Dataset** contains **5,500 audio recordings** collected during excursions to three forests in the Kenyan South Rift region:

- **Oloolua Forest** (Kajiado County)
- **Ngong Hills Forest Reserve** (Kajiado County)
- **Nyangores Forest** (Bomet County)

The primary purpose of this dataset is to address the limited availability of natural sound recordings from tropical forest ecosystems. The dataset can be used for machine learning classification tasks that detect and classify natural sounds occurring in forest environments.

### Potential Use Cases

1. Detecting human conversation in forest ecosystems.
2. Identifying Swahili-English human conversations.
3. Acoustic biodiversity and environmental monitoring.
4. Training and evaluation of audio classification models.

### Refer to the datasheet on the Zenodo record for more details regarding collection and use of the dataset.
---

## Dataset Structure

1. Audio files are provided in **WAV format** using the naming convention:
   ```text
   ZOOM0XXX_X_16k.wav
   ```
2. Clip durations range from **0.07 seconds to 5.02 seconds**.
3. The dataset README (Dataset_Description.xlsx) contains two annotation tabs:
   - **Human-Annotated**
   - **ML-Annotated-People-Talking**
4. The **Human-Annotated** subset contains **3,164** sounds labelled by human annotators.
5. The **ML-Annotated-People-Talking** subset contains **2,336** sounds of human conversation detected by a pre-trained model and subsequently validated by human annotators.
6. Details of the ML annotation workflow are available at:
   - https://github.com/SCES-Makerspace-Lab/Auto-Sound-Annotation-2026

---

## Sound Labels

The dataset contains the following labels as done conventionally by the research team:

| Label | Count |
|---------|------:|
| Axe | 90 |
| Bag | 12 |
| Bag Zip | 22 |
| Birds | 213 |
| car passing | 1 |
| Chainsaw | 129 |
| Child | 2 |
| cow mooing | 2 |
| Creaking | 1 |
| Crickets | 3 |
| Cutting wood | 7 |
| Distant talking | 15 |
| disturbance noise | 6 |
| Drone | 5 |
| Engine | 3 |
| Flowing water | 21 |
| Livestock | 26 |
| Machine beeping | 1 |
| Machine moving | 6 |
| Motorcycle | 5 |
| Moving object | 8 |
| Moving vehicle | 6 |
| Objects moving | 46 |
| Objects moving in water | 11 |
| passing vehicle | 3 |
| people talking | 39 |
| people talking and laughing | 1 |
| person laughing | 2 |
| Person moving | 9 |
| Person singing | 3 |
| Person talking | 3448 |
| person talking over disturbance noise | 1 |
| Person walking | 379 |
| Phone | 4 |
| Plastic wrapper | 11 |
| pouring water | 4 |
| radio | 1 |
| Shuffling | 13 |
| sloshing water | 4 |
| sloshing water and talking | 1 |
| Steel banging | 1 |
| Vehicle | 703 |
| White Noise | 1 |
| Wind | 231 |
| **SUM** | **5500** |

---

## Citation and Attribution

When using this dataset, please cite the dataset record available on Zenodo: https://zenodo.org/records/21443074

#### Any questions/issues are more than welcome: please write to Allan Vikiru (avikiru@strathmore.edu) with your query!
