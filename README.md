# DP Epsilon Tuner

🚀 A research framework for exploring the **privacy–utility tradeoff** in differential privacy.  
This project automatically **tunes the privacy budget (ε)** in synthetic data generators (DP-TVAE, DP-CTGAN) and finds the **smallest ε** that preserves downstream utility.

---

## ✨ Features
- **Auto-tuner for ε**: searches over DP noise multipliers (σ) and uses a privacy accountant to compute ε.  
- **Synthetic data generators**: supports DP-TVAE and DP-CTGAN (with DP-SGD).  
- **Utility evaluation**: trains downstream classifiers on synthetic data and evaluates AUROC on real test sets.  
- **Privacy probes**: optional membership-inference risk checks.  
- **Compliance-ready**: results can be mapped to HIPAA de-identification and regulatory reporting.

---

## 📊 Research Problem
Hospitals and AI centers struggle to select a **differential privacy budget (ε)** when training models with private patient data.  
- Too small ε → strong privacy but poor utility.  
- Too large ε → weak privacy but good accuracy.  

This project addresses that by **automatically finding ε***:
\[
\min \epsilon \quad \text{s.t. utility (AUROC)} \geq \tau
\]

---

## ⚙️ Installation

Clone the repo:
```bash
git clone https://github.com/your-username/dp-epsilon-tuner.git
cd dp-epsilon-tuner
