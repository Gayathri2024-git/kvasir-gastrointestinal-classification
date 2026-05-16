# Gastrointestinal Disease Classification — StFX 2025

A hybrid deep learning and classical machine learning study for 
multiclass gastrointestinal disease classification using the 
Kvasir endoscopy dataset. Developed as part of **CSCI 546** at 
St. Francis Xavier University.

---

## 👩‍🎓 Authors

**Gayathri Thirumoorthi** | x2024gpp@stfx.ca  
**Pradeep Reddy Lopinti** | x2024gqa@stfx.ca  
**Daniel Attah** | x2025ckz@stfx.ca  

Department of Computer Science, St. Francis Xavier University,  
Nova Scotia, Canada

---

## 📄 Project Overview

Gastrointestinal diseases such as colorectal cancer, ulcerative 
colitis, and esophagitis represent major global health challenges. 
This study evaluates deep learning and classical machine learning 
strategies for automated classification of gastrointestinal 
endoscopic images using the Kvasir dataset.

A pretrained **DenseNet121** model was used as the backbone for 
feature extraction and fine-tuning. Deep feature embeddings were 
then used as input to classical ML models via the **df-analyze** 
automated ML framework.

**Dataset:** Kvasir (8-class gastrointestinal endoscopy images)  
**Backbone Model:** DenseNet121 (pretrained)  
**Framework:** df-analyze (automated ML pipeline)

---

## 🔬 Methodology

### Dataset — Kvasir (8 Classes)
| Class |
|---|
| Dyed-lifted-polyps |
| Dyed-resection-margins |
| Esophagitis |
| Normal-cecum |
| Normal-pylorus |
| Normal-z-line |
| Polyps |
| Ulcerative-colitis |

### Transfer Learning Strategies
| Strategy | Description |
|---|---|
| TTL (Transfer Learning) | Pretrained model as fixed feature extractor |
| PEFT (Parameter-Efficient Fine-Tuning) | Selective layer updates |
| FFT (Full Fine-Tuning) | All layers updated for full adaptation |

### Classical ML Models (on deep embeddings)
`SGD` `Logistic Regression` `Random Forest` 
`LightGBM` `K-Nearest Neighbors` `GANDALF` `Dummy Baseline`

### Feature Selection Methods
- Association-based selection
- Prediction-based selection
- Embedded linear selection
- Wrapper-based selection
- No selection (baseline)

### Ensemble Approach
Embeddings from FFT, TTL, and PEFT were concatenated to form 
a richer feature representation for downstream classification.

### Evaluation Metrics
- Accuracy, Balanced Accuracy
- AUROC (Area Under ROC Curve)
- F1-Score, Sensitivity, Specificity
- Positive & Negative Predictive Value
- Confidence vs Adaptive Expected Error Rate

---

## 📊 Key Results

### Deep Learning — Fine-tuned DenseNet121
| Metric | Value |
|---|---|
| Accuracy | 89.2% |
| AUROC | 0.994 |
| F1-Score | 0.891 |

### Best Classical ML — SGD on Deep Embeddings
| Metric | Value |
|---|---|
| Accuracy | **93.6%** |
| AUROC | **0.996** |
| Balanced Accuracy | 0.936 |
| F1-Score | 0.936 |
| Specificity | 0.991 |

### Top df-analyze Model Configurations
| Model | Selection | Accuracy | AUROC |
|---|---|---|---|
| SGD | None | 0.936 | 0.996 |
| SGD | Assoc | 0.934 | 0.996 |
| SGD | Embed_linear | 0.934 | 0.996 |
| LR | Embed_lgbm | 0.930 | 0.996 |
| LightGBM | Assoc | 0.928 | 0.995 |

### Hypothesis Outcomes
| Hypothesis | Outcome |
|---|---|
| H1: FFT outperforms feature extraction | ✅ Supported |
| H2: PEFT achieves competitive performance | ⚠️ Partially Supported |
| H3: Classical ML won't outperform deep learning | ❌ Not Supported |
| H4: Different strategies produce different outcomes | ✅ Supported |

**Key Finding:** Classical ML models trained on deep feature 
embeddings (93.6%) outperformed standalone fine-tuned deep 
learning (89.2%), exceeding the typical 85–90% accuracy 
range reported in literature.

---

## 🛠️ Technologies & Tools

| Category | Tools / Libraries |
|---|---|
| Language | Python 3 |
| Deep Learning | DenseNet121, PyTorch / TensorFlow |
| ML Framework | df-analyze |
| Models | SGD, LR, Random Forest, LightGBM, KNN, GANDALF |
| Validation | Holdout, 5-Fold Cross-Validation |
| Evaluation | scikit-learn, AUROC, Adaptive Error Estimation |
| Dataset | Kvasir (Simula Research Laboratory) |

---

## 📂 Repository Structure
