# ​​FusNeoPred​​  
## Introduction
Fusion-derived neoantigens represent superior therapeutic targets for precision immunotherapy due to their high immunogenicity and clonality; however, their accurate identification is hindered by the scarcity of labeled data and the gap between in silico predictions and biological reality. Here, we present FusNeoPred, a weakly supervised learning framework that innovatively integrates Nanopore long-read sequencing with digital twins. By superimposing precise transcriptomic structures onto the operational digital twinning, our approach moves beyond static prediction. The framework employs a multi-module scoring system, featuring a customized Sigmoid transformation for MHC binding metrics to better evaluate physiological immunogenicity. A defining innovation of our study is the closed-loop model calibration strategy utilizing real experimental data. We established a feedback mechanism where the computational model is rigorously fine-tuned using ground-truth results derived from peptide synthesis and IFN-γ ELISpot assays in humanized mice. This iterative assimilation of wet-lab data continuously corrects the virtual parameters to align with biological complexity. In summary, the FusNeoPred framework constructed in this study effectively identifies high-confidence fusion neoantigens, offering a scalable and biologically calibrated solution for the development of next-generation cancer vaccines.  
<img width="2269" height="1315" alt="graphical abstract-01-01" src="https://github.com/user-attachments/assets/f01f6e87-88b3-4953-bf26-728a838b155c" />
## Installation  

You can install just the base python(v3.8) packages, include: 
- pandas
- numpy
- torch
- scikit-learn
- matplotlib
- joblib

We recommend creating the environment and installing it with conda:  

1. Create conda envireoment  
  
```bash
conda create -n FusNeoPred python=3.8.5
```

2. Activate conda  

```bash  
conda activate FusNeoPred
```

3. Install python packages  

```bash
pip install pandas==2.0.3 numpy==1.24.3 torch==2.4.1 scikit-learn==1.3.2 matplotlib==3.7.5 joblib==1.4.2
```

## Step1: Data preparation
Run the Python script to generate preparation data.  
```bash
python combine_peptide_data.py
```
After generating S5_combined_complete.tsv,  Link S5_combined_complete.tsv with SumScore file from Step1 folder.  
```bash
awk 'BEGIN{FS=OFS="\t"}NR==FNR{a[$1]=$2}NR>FNR{print $0,a[$2]}' SumScore S5_combined_complete.tsv > S5_combined_complete_score.tsv
```

## Step2: Training
This step requires the following input files:  
- S5_combined_complete_score.tsv  
​A tab-separated file with a header, containing at least 75 columns where columns 3-74 are features and column 75 is the target score for pre-training.  
- Peplist_Score  
​A tab-separated file without a header, following the same column structure as the main file, and must contain both positive and negative samples for fine-tuning.  

Then run training process：  
```bash
python MLP_251230_tuneplot.py
```

The script will generate the following output files:  
- training_log.txt  
A log file recording detailed training progress, including losses for each epoch in both pre-training and fine-tuning phases.  
- pretrain_curve.pdf  
A plot visualizing the pre-training loss curves (train, validation, and test MSE).  
- final_model_tuned.pth  
The saved PyTorch model weights after fine-tuning (if fine-tuning is performed).  
- scaler.pkl  
The fitted StandardScaler object saved via joblib for consistent data preprocessing.  
