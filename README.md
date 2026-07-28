# ​​FusNeoPred​​  
## Introduction
Fusion-derived neoantigens are ideal targets for precision immunotherapy because of their high immunogenicity and clonality. However, their reliable identification is limited by the scarcity of labeled data and the gap between in silico predictions and biological reality. We developed FusNeoPred, a biologically calibrated deep learning framework inspired by digital twins. Integrating Nanopore long read sequencing, it employs a multimodule scoring system featuring a customized Sigmoid transformation for MHC binding affinity to better reflect physiological immunogenicity. The feature of our framework is the closed loop calibration strategy, in which a multilayer perceptron model is first pretrained using expert rule derived pseudolabels, and is subsequently fine tuned iteratively with experimental feedback obtained from peptide synthesis and IFN-γ ELISpot assays conducted in humanized HLA-A2.1 transgenic mice. Profiling colorectal cancer samples, FusNeoPred initially identified 19,651 candidate neoantigens from 32 high confidence fusion transcripts. In vivo validation confirmed that top predicted fusion neoantigens elicited robust antigen specific T cell responses and higher IFN-γ secretion than controls. Integrating this wet-lab data via a pairwise margin ranking loss allowed the model to rapidly adapt to the true ranking order without catastrophic forgetting. The calibrated model demonstrated high robustness, retaining 80%–92% of top tier candidates while successfully reprioritizing peptides by capturing nonlinear immunogenic features missed by static rules. In summary, FusNeoPred effectively identifies high confidence fusion neoantigens, providing a scalable and biologically calibrated platform for developing cancer vaccines.  

<img width="1114" height="646" alt="graphical abstract-01-01" src="https://github.com/user-attachments/assets/91b79b7c-094f-484b-a7c0-8dcfc4f55d29" />




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

<img width="3071" height="1713" alt="Figure 1_v2_0723-01" src="https://github.com/user-attachments/assets/e56e987f-02a0-4b58-b433-828bc04ea849" />


Training process：  
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


## Step3: Prediction
This step requires Step2 model files, final_model_tuned.pth and scaler.pkl.  

```bash
python predict.py S5_combined_complete.tsv S5_score_predict.tsv
```
The input file is S5_combined_complete.tsv, and the output score file is S5_score_predict.tsv.  

## Step4: Drawing
Figures in the manuscript comparing model performance before and after ranking were generated using the scripts in the Step4 folder.  

<img width="966" height="344" alt="image" src="https://github.com/user-attachments/assets/65ee2466-291c-4282-b828-715581a41f27" />

## Step5: Toxicity and allergenicity prediction
After obtaining the candidate neoantigen peptides, their toxicity was evaluated using ToxinPred 3.0 (https://webs.iiitd.edu.in/raghava/toxinpred3/) and CSM-Toxin (https://biosig.lab.uq.edu.au/csm_toxin/), while their allergenicity was assessed using AllerTOP v2.1 (https://www.ddg-pharmfac.net/allertop_v2/) and AlgPred 2.0 (https://webs.iiitd.edu.in/raghava/algpred2/).

The prediction output summary was like:  
|SequenceID|Sequence|ToxinPred 3.0|CSM-Toxin| AltterTop v2.1| AlgPred 2.0 |
|----------|--------|----------|-----------|-----------|-----------|
|1         |ALPAFVALL |Non-Toxin|Toxic|NON-ALLERGEN|ALLERGEN|
|2         |LLLLSPWPL|Non-Toxin|Non-Toxin|NON-ALLERGEN|ALLERGEN|
|3         |GQFSAVHPNV|Non-Toxin|Non-Toxin|NON-ALLERGEN|ALLERGEN|




