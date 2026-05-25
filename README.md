===============================================================================
                    PASS-01 MULTIMODAL VALIDATION PIPELINE                    
===============================================================================

DESCRIPTION
-----------
This repository contains the standardized framework for the MULTIPL/PASS-01 
multimodal machine learning architecture. The pipeline evaluates unimodal and 
fused computational estimators designed to predict Differential Treatment 
Effects (DTE) and therapeutic endpoints in Pancreatic Ductal Adenocarcinoma 
(PDAC) cohorts.


SUPPORTED DATA MODALITIES
-------------------------
The framework supports evaluation across four distinct data modalities:
  [1] CLINICAL         [2] GENOMIC (DNA)   
  [3] TRANSCRIPTOMIC (RNA)  [4] HISTOPATHOLOGY (WSI)


===============================================================================
                       IMPORTANT: DATA PRIVACY NOTICE                          
===============================================================================
CRITICAL: Raw and processed genomic/clinical cohorts from the PASS-01 trial 
are highly protected patient data and are NOT uploaded to this repository.
===============================================================================


===============================================================================
                          CODE REPOSITORY STRUCTURE                            
===============================================================================

multipl-pass01/
|
+-- .gitignore
+-- README.md
+-- environment.yml
|
+-- src/
|   |
|   +-- training/                 [PART 1: Core Training Modules]
|   |   +-- preprocess.py
|   |   +-- train_unimodal.py
|   |   +-- train_early_fusion.py
|   |   +-- train_late_fusion.py
|   |
|   +-- validation/               [PART 2: Independent Validation Suite]
|       +-- validate_pipelines.py
|
===============================================================================
                          ENVIRONMENT SETUP                                    
===============================================================================

Navigate to the validation subdirectory and build the Conda environment:

  cd src/validation/
  conda env create -f environment.yml
  conda activate TabPFN


===============================================================================
                       CRITICAL PIPELINE EXECUTION ORDER                       
===============================================================================

Operational flow MUST follow this chronological sequence:

STEP 1: MODEL TRAINING
----------------------
You must run the training pipeline scripts inside `src/training/` to generate 
and freeze your model checkpoints before attempting any evaluation.

STEP 2: COHORT VALIDATION
-------------------------
Once checkpoints exist, execute the main evaluation wrapper:
  python src/validation/validate_pipelines.py

MODALITY DEPENDENCY RULES (Applies to both Training and Validation):
-------------------------------------------------------------------
1. UNIMODAL execution MUST occur first to generate base predictions.
2. EARLY FUSION and LATE FUSION passes can be run in any order once 
   Unimodal baselines are established.

===============================================================================