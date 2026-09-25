# MultiEYE: OCT-Assisted Retinal Disease Recognition

An implementation of **OCT-assisted conceptual knowledge distillation for retinal disease classification from fundus images**, based on the MultiEYE framework proposed by Wang et al.

The project explores how knowledge learned from **Optical Coherence Tomography (OCT)** images can be transferred to a **fundus-image model**, allowing the final model to perform retinal disease classification using fundus images alone.

> **Reference:** Wang et al., *MultiEYE: Dataset and Benchmark for OCT-Enhanced Retinal Disease Recognition From Fundus Images*, IEEE Transactions on Medical Imaging, 2025.

---

## 📌 Overview

Traditional multimodal retinal disease classification approaches often require paired fundus and OCT images during both training and testing.

The MultiEYE approach uses a different setting:

* **OCT images** are used during training to provide additional disease-related knowledge.
* **Fundus images** are used to train the target model.
* The OCT model acts as a **teacher**.
* The fundus model acts as a **student**.
* During inference, only the **fundus model** is required.

The original MultiEYE work introduces **OCT-CoDA (OCT-assisted Conceptual Distillation Approach)**, where disease-related concepts are used as an intermediate representation for transferring knowledge from OCT to fundus images.

---

## 🎯 Project Objective

The objective of this implementation is to reproduce the core OCT-assisted knowledge distillation pipeline and investigate whether OCT-derived knowledge can improve retinal disease recognition from fundus photographs.

The implementation focuses on:

1. Training an OCT teacher model.
2. Training a fundus student model.
3. Generating image-concept similarity representations.
4. Transferring knowledge using:

   * Global Prototypical Distillation (GPD)
   * Local Contrastive Distillation (LCD)
5. Handling class imbalance using class-weighted cross-entropy.
6. Evaluating the final fundus model using multiple classification metrics.

---

## 🧠 Methodology

### 1. Concept-Based Representation

The model uses disease-related textual concepts and a vision-language model to calculate the similarity between an image and the concepts.

For each image:

```text
Image
  ↓
Vision Encoder
  ↓
Image Embedding
  ↓
Concept Similarity
  ↓
Concept Classifier
  ↓
Disease Prediction
```

The original paper generates disease concepts using an LLM and uses a vision-language model to relate image features with those concepts.

---

### 2. OCT Teacher Model

The first stage trains a teacher model using OCT images.

```text
OCT Image
    ↓
FLAIR Vision Encoder
    ↓
Concept Similarities
    ↓
Concept Classifier
    ↓
Disease Prediction
```

The trained OCT model is then frozen and used as the teacher during student training.

---

### 3. Fundus Student Model

The second stage trains the target fundus model.

```text
                 ┌─────────────────┐
                 │   OCT Teacher   │
                 └────────┬────────┘
                          │
                    OCT Knowledge
                          │
                          ▼
Fundus Image → Student Model → Disease Prediction
                    ▲
                    │
              GPD + LCD Loss
```

The OCT branch is only required during training. At inference time, the OCT branch is discarded and the fundus model performs the prediction.

---

## 🔄 Knowledge Distillation

### Global Prototypical Distillation (GPD)

GPD aligns the disease-level concept representations between the OCT teacher and fundus student.

For each disease class, a prototype is calculated from the concept similarities of the samples.

The loss minimizes the distance between the OCT and fundus class prototypes.

```text
OCT Concept Prototype
          │
          │
          ▼
     GPD Loss
          ▲
          │
          │
Fundus Concept Prototype
```

This transfers generalized disease-level information from the OCT modality to the fundus modality.

---

### Local Contrastive Distillation (LCD)

LCD operates at the sample level.

Samples belonging to the same disease class are treated as positive pairs, while samples from different classes act as negatives.

This helps the student model:

* Learn sample-level disease characteristics from OCT.
* Preserve useful fundus-specific information.
* Bring representations of the same disease closer together.

---

### Total Training Loss

The student model uses:

```text
Total Loss
    =
Classification Loss
    +
α × GPD Loss
    +
β × LCD Loss
```

In this implementation:

```text
α = 0.6
β = 0.05
τ = 10.0
```

These values follow the configuration used in the reference implementation/paper.

---

## 🩺 Disease Classes

The MultiEYE benchmark contains nine disease categories:

| Class | Description                                 |
| ----- | ------------------------------------------- |
| 0     | Normal                                      |
| 1     | Dry Age-related Macular Degeneration (dAMD) |
| 2     | Central Serous Chorioretinopathy (CSC)      |
| 3     | Diabetic Retinopathy (DR)                   |
| 4     | Glaucoma (GLC)                              |
| 5     | Macular Epiretinal Membrane (MEM)           |
| 6     | Myopia (MYO)                                |
| 7     | Retinal Vein Occlusion (RVO)                |
| 8     | Wet Age-related Macular Degeneration (wAMD) |

The original MultiEYE dataset contains **58,036 fundus photographs and 45,923 OCT B-scans** across these nine classes.

---

## 🏗️ Model Configuration

### Backbone

This implementation uses:

* **FLAIR**
* ResNet-based vision encoder
* Vision-language concept representation
* Frozen text encoder
* Trainable image encoder and concept classifier

The notebook loads the FLAIR implementation and uses the FLAIR model for the concept-based classification pipeline.

### Image Processing

Fundus preprocessing includes:

* Resize
* Random crop
* Horizontal flip
* Vertical flip
* Rotation
* Color jitter
* Normalization

OCT preprocessing includes:

* Resize
* Horizontal flip
* Rotation
* Normalization

The original paper similarly applies preprocessing and augmentation to both modalities.

---

## ⚖️ Class Imbalance Handling

The implementation calculates **class weights** from the fundus training labels and uses weighted cross-entropy during training.

This is particularly useful because retinal disease datasets can contain substantially different numbers of samples across classes.

```python
class_weights = get_class_weights(fundus_train_labels)

loss = F.cross_entropy(
    logits,
    labels,
    weight=class_weights
)
```

---

## ⚙️ Training Configuration

| Parameter             |                     Value |
| --------------------- | ------------------------: |
| Number of classes     |                         9 |
| Teacher epochs        |                        80 |
| Student epochs        |                       100 |
| Batch size            |                         4 |
| Gradient accumulation |                        16 |
| Effective batch size  |                        64 |
| Learning rate         |                      1e-4 |
| Optimizer             |                     AdamW |
| Weight decay          |                      1e-4 |
| GPD weight (α)        |                       0.6 |
| LCD weight (β)        |                      0.05 |
| Temperature (τ)       |                        10 |
| Scheduler             | Warmup + Cosine Annealing |
| Gradient clipping     |                       1.0 |

The notebook uses gradient accumulation to obtain an effective batch size of 64 while keeping the per-step batch size small enough for GPU memory constraints.

---

## 📊 Evaluation Metrics

The implementation evaluates the model using:

* Precision
* Recall
* Specificity
* Precision-Recall F1
* Sensitivity-Specificity F1
* Mean Average Precision (MAP)
* Accuracy
* Cohen's Kappa

These metrics are also used in the MultiEYE paper to evaluate performance under class imbalance.

---

## 📁 Project Structure

```text
MultiEYE/
│
├── mini_proj_v4.ipynb
├── README.md
│
├── checkpoints/
│   ├── oct_teacher_final.pth
│   ├── student_best.pth
│   └── ...
│
└── results/
    ├── training_curves/
    └── evaluation_results/
```

> Dataset files and large model checkpoints are intentionally not included in this repository unless required.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Install dependencies

```bash
pip install torch torchvision
pip install transformers
pip install scikit-learn
pip install numpy pandas matplotlib
pip install pillow tqdm
```

Install the FLAIR implementation:

```bash
pip install git+https://github.com/jusiro/FLAIR.git
```

### 3. Prepare the Dataset

Download/obtain the MultiEYE dataset and organize the paths according to the notebook configuration.

Update:

```python
DATA_ROOT = "/path/to/MultiEYE/multieye_data/"
```

Also update the concept file path:

```python
concepts_path = "/path/to/concepts_raw.npy"
```

### 4. Run the Notebook

Open:

```text
mini_proj_v4.ipynb
```

and execute the cells in order.

---

## 💾 Checkpoints

The notebook supports checkpoint-based training and resuming.

Example:

```text
checkpoints/
├── teacher_latest.pth
├── teacher_scheduler.pth
├── oct_teacher_final.pth
├── student_latest.pth
├── student_best.pth
└── student_scheduler.pth
```

This allows long training runs to be resumed without starting from the beginning.

---

## 🔬 Training Pipeline

The complete workflow is:

```text
                 MultiEYE Dataset
                       │
          ┌────────────┴────────────┐
          │                         │
      OCT Images              Fundus Images
          │                         │
          ▼                         ▼
   OCT Teacher Model        Fundus Student Model
          │                         │
          │                  Classification Loss
          │                         │
          └───────┐         ┌───────┘
                  │         │
                  ▼         ▼
                 GPD + LCD
                     │
                     ▼
              Student Training
                     │
                     ▼
             Final Fundus Model
                     │
                     ▼
              Fundus Image Only
                     │
                     ▼
              Disease Prediction
```

---

## 📈 Results

The notebook evaluates the trained student model on the fundus validation set and reports the implemented MultiEYE evaluation metrics.

Example output:

```python
metrics = evaluate(
    student_model,
    fundus_val_loader,
    device,
    phase="Student Eval"
)

metrics
```

The reported metrics include:

```text
Precision
Recall
Specificity
P-R F1
S-S F1
MAP
Accuracy
Kappa
```

Add your final experimental results here after completing the final run:

| Metric      | Student Model |
| ----------- | ------------: |
| Precision   |             — |
| Recall      |             — |
| Specificity |             — |
| P-R F1      |             — |
| S-S F1      |             — |
| MAP         |             — |
| Accuracy    |             — |
| Kappa       |             — |

---

## 📚 Reference

This project is based on:

**Lehan Wang, Chongchong Qi, Chubin Ou, Lin An, Mei Jin, Xiangbin Kong, and Xiaomeng Li.**

> *MultiEYE: Dataset and Benchmark for OCT-Enhanced Retinal Disease Recognition From Fundus Images.*

IEEE Transactions on Medical Imaging, Vol. 44, No. 4, 2025.

Original research repository:

```text
https://github.com/xmed-lab/MultiEYE
```

---

## ⚠️ Disclaimer

This repository contains an academic implementation/experimental reproduction of the MultiEYE methodology.

It is intended for **research and educational purposes only** and should not be used as a standalone medical diagnostic system.

The original MultiEYE research also notes limitations related to unseen diseases and dependence on the ophthalmology knowledge contained in the pretrained foundation model.

---

## 👩‍💻 Author

**Johana Priscy John Ponraj**

B.Tech CSE — Artificial Intelligence & Data Science
SASTRA Deemed University

---

⭐ If you find this implementation useful, consider starring the repository.
