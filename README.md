# Multi-Eye — OCT-Enhanced Retinal Disease Recognition

**Multi-Eye** is a deep learning project for **retinal disease recognition from fundus images**, enhanced using knowledge learned from **Optical Coherence Tomography (OCT)** images.

The project is based on the **MultiEYE** dataset and the **OCT-assisted Conceptual Distillation Approach (OCT-CoDA)** proposed in the research work:

> **MultiEYE: Dataset and Benchmark for OCT-Enhanced Retinal Disease Recognition From Fundus Images**
> Lehan Wang, Chongchong Qi, Chubin Ou, Lin An, Mei Jin, Xiangbin Kong, Xiaomeng Li
> *IEEE Transactions on Medical Imaging, 2025*

The main idea is to use OCT images as a **teacher modality during training** and transfer disease-related knowledge to a **fundus-image student model**. During inference, only the fundus image is required.

---

## 📌 Project Overview

Fundus photography and Optical Coherence Tomography (OCT) provide complementary information for retinal disease diagnosis.

* **Fundus images** provide a comprehensive view of the retina, optic nerve, and vascular network.
* **OCT images** provide detailed information about retinal structure and thickness.

Conventional multimodal approaches generally require paired fundus and OCT images during both training and testing. However, obtaining paired multimodal medical data can be difficult.

This project follows the **OCT-enhanced disease recognition from fundus images** setting:

```text
                 Training
        ┌─────────────────────────┐
        │                         │
        ▼                         ▼
   OCT Images                Fundus Images
        │                         │
        ▼                         ▼
 OCT Teacher Model          Fundus Student Model
        │                         │
        │    Knowledge Transfer   │
        └──────────┬──────────────┘
                   ▼
             OCT-CoDA
                   │
                   ▼
          Trained Fundus Model
                   │
                   ▼
                Testing
                   │
                   ▼
          Fundus Image Only
                   │
                   ▼
          Retinal Disease
             Prediction
```

The research framework specifically uses **unpaired multimodal data during training** and requires only fundus photographs during testing.

---

# 🎯 Objectives

The main objectives of the project are:

* To perform retinal disease classification using fundus images.
* To utilize OCT images as an additional source of disease-related knowledge during training.
* To transfer knowledge from an OCT teacher model to a fundus student model.
* To use disease-related concepts as an interpretable connection between the two modalities.
* To improve fundus-based retinal disease recognition without requiring OCT images during inference.

---

# 🧠 Proposed Approach

The project follows the **OCT-assisted Conceptual Distillation Approach (OCT-CoDA)**.

OCT-CoDA consists of three major stages:

1. **LLM-based concept generation**
2. **Concept-decoupled disease classification**
3. **OCT-assisted conceptual knowledge distillation**

The proposed framework uses the relationship between **image features and disease-related concepts** to transfer useful knowledge from OCT images to fundus images.

---

## 1. LLM-Based Concept Generation

Disease-related visual concepts are generated to describe fine-grained characteristics of retinal diseases.

Instead of relying only on disease labels, the approach uses detailed disease attributes as an intermediate representation.

The research uses an LLM-based process to generate disease concepts and applies a Chain-of-Thought style prompting strategy to consider different retinal regions and disease characteristics.

Concepts can describe characteristics such as:

* Color
* Shape
* Location
* Retinal abnormalities
* Structural changes
* Disease-specific visual characteristics

These concepts provide an interpretable bridge between the OCT and fundus modalities.

---

# 2. Concept-Decoupled Network

The image encoder extracts visual features from each image.

These image features are compared with the corresponding concept embeddings.

```text
Image
  │
  ▼
Image Encoder
  │
  ▼
Image Features
  │
  ├───────────────┐
  │               │
  ▼               ▼
Concept Embeddings
  │
  ▼
Image-Concept Similarity
  │
  ▼
Concept Classifier
  │
  ▼
Disease Prediction
```

The image-concept similarity matrix is used as the input to the final fully connected classification layer.

---

# 3. OCT Teacher and Fundus Student

The framework uses two models:

### OCT Teacher

The OCT model learns disease-related information from OCT scans.

### Fundus Student

The fundus model learns to perform retinal disease classification using fundus photographs while receiving additional knowledge from the OCT teacher.

```text
             OCT Image
                 │
                 ▼
          OCT Image Encoder
                 │
                 ▼
          OCT Concept Features
                 │
                 │
                 │ Knowledge
                 │ Distillation
                 ▼
          Fundus Student
                 ▲
                 │
          Fundus Image
```

The OCT model acts as the **teacher**, while the fundus model acts as the **student**.

---

# 🔄 Knowledge Distillation

The proposed OCT-CoDA framework consists of two major knowledge-distillation components:

* **Global Prototypical Distillation (GPD)**
* **Local Contrastive Distillation (LCD)**

These components transfer disease-related knowledge from the OCT modality to the fundus modality.

---

## Global Prototypical Distillation

**GPD** transfers global disease-level information.

The method constructs class-level prototypes from the concept representations and aligns the OCT teacher representation with the fundus student representation.

```text
OCT Images
     │
     ▼
OCT Concept Representation
     │
     ▼
Class Prototypes
     │
     │
     ▼
   GPD Loss
     ▲
     │
     │
Fundus Concept Representation
     │
     ▼
Fundus Class Prototypes
```

This encourages the fundus student to learn disease-level characteristics captured by the OCT teacher.

---

## Local Contrastive Distillation

**LCD** performs sample-level knowledge transfer.

It encourages samples belonging to the same disease category to have similar representations while separating representations belonging to different disease categories.

```text
OCT Sample ──────────────┐
                         │
                         ▼
                    Contrastive
                       Loss
                         ▲
                         │
Fundus Sample ───────────┘
```

This allows the fundus student to learn finer disease-related characteristics from the OCT modality.

---

# 🩺 Disease Categories

The MultiEYE dataset contains **nine retinal disease categories**:

| No. | Disease                                     |
| --: | ------------------------------------------- |
|   0 | Normal                                      |
|   1 | Dry Age-related Macular Degeneration (dAMD) |
|   2 | Central Serous Chorioretinopathy (CSC)      |
|   3 | Diabetic Retinopathy (DR)                   |
|   4 | Glaucoma (GLC)                              |
|   5 | Macular Epiretinal Membrane (MEM)           |
|   6 | Myopia (MYO)                                |
|   7 | Retinal Vein Occlusion (RVO)                |
|   8 | Wet Age-related Macular Degeneration (wAMD) |

The dataset construction process retained samples with a single disease label and divided the data into training, validation, and test sets while ensuring that images from the same patient do not appear across different subsets.

---

# 📊 MultiEYE Dataset

The MultiEYE benchmark contains:

* **58,036 fundus photographs**
* **45,923 OCT B-scans**
* **9 disease categories**

The dataset combines public datasets with in-house data and was designed specifically for OCT-enhanced retinal disease recognition from fundus images.

The dataset uses **unpaired multimodal data**, meaning the OCT and fundus images do not necessarily belong to the same patient, but share the same disease label space.

---

# 🏗️ Model Architecture

The research evaluates the approach using vision-language model backbones including:

* **CLIP**
* **FLAIR**

The image encoder is based on **ResNet-50**.

For FLAIR, images are resized to **512 × 512**, while CLIP uses **224 × 224** images.

The project implementation uses the concept-based vision-language approach to connect retinal images with disease-related concepts.

---

# 🛠️ Technologies

The project involves the following technologies and concepts:

* Python
* PyTorch
* Deep Learning
* Computer Vision
* Medical Image Analysis
* Vision-Language Models
* FLAIR
* ResNet-50
* Knowledge Distillation
* Contrastive Learning
* Concept Bottleneck / Concept-based Learning
* Large Language Models
* OCT Image Processing
* Fundus Image Processing
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

---

# 🖼️ Image Preprocessing

The research preprocessing pipeline includes:

### Fundus Images

* Contrast-Limited Adaptive Histogram Equalization (CLAHE)
* Resizing
* Random cropping
* Flipping
* Rotation
* Contrast adjustment
* Saturation adjustment
* Brightness adjustment

### OCT Images

* Median filtering
* Resizing
* Random cropping
* Flipping
* Rotation
* Image augmentation

These preprocessing and augmentation operations are applied to improve model training and maintain consistency between experiments.

---

# ⚙️ Training Configuration

The reported implementation uses:

| Parameter                |                   Value |
| ------------------------ | ----------------------: |
| Backbone                 |               ResNet-50 |
| Vision-language backbone |            CLIP / FLAIR |
| Optimizer                |                   AdamW |
| Learning Rate            |                  `1e-4` |
| Batch Size               |                    `64` |
| Learning Rate Decay      |        Cosine Annealing |
| Temperature (τ)          |                    `10` |
| GPD Weight (α)           |                   `0.6` |
| LCD Weight (β)           |                  `0.05` |
| GPU                      | NVIDIA GeForce RTX 3090 |

These values are reported in the MultiEYE research paper.

---

# 📈 Evaluation Metrics

The model is evaluated using eight metrics:

1. **Precision**
2. **Recall**
3. **Specificity**
4. **Precision-Recall F1**
5. **Sensitivity-Specificity F1**
6. **Mean Average Precision (MAP)**
7. **Accuracy**
8. **Cohen's Kappa**

These metrics provide a broader evaluation of classification performance, particularly for class-imbalanced medical datasets.

---

# 🔬 Experimental Comparison

The MultiEYE research evaluates multiple approaches, including:

### Fundus Model

A fundus feature extractor followed by a linear classifier.

### Fundus + Concept

A concept-decoupled fundus model using a fixed text encoder, trainable vision encoder, and concept classifier.

### FDDM

A knowledge-distillation approach adapted to transfer knowledge from OCT to fundus in the MultiEYE setting.

### OCT-CoDA

The proposed OCT-assisted conceptual distillation framework using:

* Disease concepts
* Global Prototypical Distillation
* Local Contrastive Distillation

---

# 🔍 Why Conceptual Distillation?

Directly transferring image features can also transfer information that is not useful for disease recognition, such as:

* Background information
* Disease-irrelevant features
* Modality-specific noise

OCT-CoDA instead uses **disease-related concepts** as an intermediate representation.

This makes the knowledge-transfer process more controlled and interpretable.

---

# 🚀 Inference

One of the key advantages of this approach is that OCT images are **not required during inference**.

The final workflow is:

```text
             Fundus Image
                   │
                   ▼
             Image Encoder
                   │
                   ▼
          Image-Concept Similarity
                   │
                   ▼
           Concept Classifier
                   │
                   ▼
          Disease Prediction
```

This allows the trained model to perform retinal disease recognition using only a fundus photograph.

---

# 📁 Repository Structure

```text
Multi-Eye/
│
├── README.md
│
├── mini_proj_v4.ipynb
│
├── results/
│   └── ...
│
├── checkpoints/
│   └── ...
│
└── data/
    └── ...
```

> Large datasets and model checkpoints should generally not be committed directly to GitHub. Configure the dataset paths locally before running the notebook.

---

# ▶️ How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/JOHANAPRISCY05/Multi-Eye.git
cd Multi-Eye
```

## 2. Install Dependencies

```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install matplotlib pillow tqdm
```

Install any additional dependencies required by the FLAIR implementation used by the notebook.

## 3. Prepare the Dataset

Download and prepare the MultiEYE dataset separately.

Update the dataset paths in the notebook to point to your local dataset location.

For example:

```python
DATA_ROOT = "/path/to/MultiEYE/"
```

## 4. Run the Notebook

Open:

```text
mini_proj_v4.ipynb
```

and execute the cells sequentially.

---

# 📌 Important Note

This repository represents an **academic implementation/study based on the MultiEYE research work**.

The underlying OCT-CoDA methodology, MultiEYE dataset, and associated research concepts belong to the original authors.

This project should not be represented as the original publication or as an independent invention of the OCT-CoDA methodology.

---

# ⚠️ Medical Disclaimer

This project is intended for **academic and research purposes only**.

The model is not a clinically validated diagnostic system and should not be used to make medical decisions or replace professional ophthalmological examination.

---

# 📚 Reference

**Lehan Wang, Chongchong Qi, Chubin Ou, Lin An, Mei Jin, Xiangbin Kong, and Xiaomeng Li.**

**“MultiEYE: Dataset and Benchmark for OCT-Enhanced Retinal Disease Recognition From Fundus Images.”**

*IEEE Transactions on Medical Imaging, Volume 44, Issue 4, April 2025.*

DOI: `10.1109/TMI.2024.3518067`

The paper describes the MultiEYE dataset and OCT-assisted Conceptual Distillation Approach (OCT-CoDA).

### Original Research Repository

https://github.com/xmed-lab/MultiEYE

---

# 👩‍💻 Author

**Johana Priscy John Ponraj**

B.Tech Computer Science and Engineering
Specialization: Artificial Intelligence & Data Science
SASTRA Deemed University

---

⭐ **If you find this project useful, consider giving the repository a star.**
