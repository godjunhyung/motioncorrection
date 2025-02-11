TITLE: Embedding-Based MRI Quality Assessment for Adaptive Thresholding and Interactive Retrieval
       — Focusing on Motion Artifact across Multiple Sequences

AUTHOR NAMES & AFFILIATIONS:
- [Author 1 Name], [Affiliation]
- [Author 2 Name], [Affiliation]
- [Author 3 Name], [Affiliation]
(Corresponding author email: xxxx@xxxx.xxx)

ABSTRACT
A brief overview (150–250 words):
- Problem: MRI quality significantly impacts diagnostic confidence, but existing no-reference IQA methods often produce a single scalar score with limited adaptability.
- Goal: We propose an embedding-based approach to capture richer quality representations and to support adaptive thresholding and retrieval in clinical workflows. 
- Method: We specifically focus on **motion artifact**, generating a simulated dataset spanning multiple MRI sequences (e.g., T1, T2) with varying levels of motion corruption. Using a single FR-IQM (weak label), we first perform metric learning on the simulation data. We then use a small real MRI dataset with expert (strong) labels and pairwise ranking to fine-tune the embedding for clinical alignment.
- Results: Our experiments demonstrate that the proposed method yields robust performance across different scan sequences under motion artifact, providing an interactive interface for dynamic quality thresholding and retrieval.
- Contribution: This work bridges no-reference IQA with image retrieval concepts, enabling clinicians to adaptively set quality thresholds and efficiently search for similar-quality images across multiple MRI sequences affected by motion.

KEYWORDS:
- No-Reference IQA, MRI Quality Assessment, Motion Artifact, Metric Learning, Adaptive Thresholding, Multi-sequence MRI

--------------------------------------------------
1. INTRODUCTION
--------------------------------------------------
1.1 Motivation
- The clinical importance of MRI quality assessment (IQA).
- **Motion artifact** is among the most common and problematic issues in MRI, leading to diagnostic uncertainty.
- MRI systems also offer multiple scan types (T1, T2, etc.), each requiring robust quality evaluation when motion occurs.

1.2 Problem Statement
- Difficulty of obtaining large-scale annotated MRI data (costly expert labeling).
- Need for adaptive or context-dependent quality thresholds in different diagnostic scenarios.
- Ensuring a single IQA model can handle **multiple scan protocols** (T1, T2, etc.) under motion artifact conditions.

1.3 Proposed Approach
- Embed MRI images into a learned metric space that reflects overall quality.
- Use simulation data + a single FR-IQM score as a weak label for initial metric learning.
  - **Focus on motion artifact simulation** across multiple sequences.
- Fine-tune with a small real dataset (expert labels) in a pairwise manner to ensure clinical alignment.
- Demonstrate generalization to **various MRI sequences** affected by motion artifact through embedding-based retrieval and thresholding.

1.4 Contributions
1) **Embedding-Based IQA** rather than single-value predictions.  
2) **Weak-to-Strong Label Transfer** using simulation-based FR-IQM followed by pairwise fine-tuning on real labels.  
3) **Motion Artifact Handling** across multiple MRI sequences.  
4) **Interactive Retrieval** with reference or pivot images for dynamic threshold adaptation.

--------------------------------------------------
2. RELATED WORK
--------------------------------------------------
2.1 No-Reference IQA Methods
- Brief survey: classical vs. deep learning approaches, ranking-based IQA, medical imaging IQA challenges.

2.2 Motion Artifact Simulation & FR-IQM in MRI
- Summarize how synthetic data with motion corruption can be generated.
- Using a full-reference metric (e.g., SSIM, VIF, etc.) to approximate quality for simulated images.

2.3 Metric Learning and Image Retrieval
- Highlight how metric learning has driven advances in retrieval tasks and its potential for IQA.
- Emphasize the advantage of an embedding space in representing motion degradation levels.

2.4 Adaptive Thresholding / Interactive Approaches
- Prior methods suggesting dynamic or user-defined thresholds; existing frameworks for flexible image quality interpretation.
- Handling multi-sequence MRI under motion artifact with an interactive or user-driven interface.

--------------------------------------------------
3. METHODS
--------------------------------------------------
3.1 Overview of the Proposed Pipeline
- Flowchart or diagram of the entire process:
  1) **Simulation Data & Single FR-IQM** → Metric learning  
  2) **Small Real Dataset & Expert Labels** → Pairwise Fine-tuning  
  3) **Inference** → Embedding-based scoring + Interactive threshold/retrieval  
- Emphasize **inclusion of multiple sequence types** (T1, T2, etc.) in the simulation or real data, all with motion artifact variations.

3.2 Metric Learning with a Single FR-IQM (Weak Label)
3.2.1 Data Generation
- **How motion artifact** is simulated across different MRI sequences (T1, T2, etc.).
- Each simulated image labeled by **one** FR-IQM score (weak label).

3.2.2 Network Architecture
- Base CNN for feature extraction (Encoder).
- A regression or ranking head to predict (or align) the FR-IQM score in the embedding.

3.2.3 Loss Function
- A loss function (e.g., MSE, L1, or a ranking loss) that aligns the embedding with the FR-IQM weak label.
- Any weighting or regularization strategies to improve stability given multiple sequences.

3.3 Pairwise Fine-Tuning with Strong Labels
3.3.1 Expert Labels
- Small real dataset with 5-level or continuous quality annotations.
- Possibly includes different sequences in real scans, but primarily focusing on motion artifact.
- Conversion to pairwise constraints (“image A > image B in quality”).

3.3.2 Loss for Pairwise Ranking
- Contrastive / Triplet / Ranking loss to align the learned embedding with clinical judgments.
- Possibly partial fine-tuning of the encoder or entire network.

3.3.3 Domain Alignment
- Optional: mention any domain adaptation if used (adversarial alignment, etc.).
- Additional considerations for **multi-sequence** distribution alignment in real data.

3.4 Embedding-Based IQA & Retrieval
3.4.1 Final Embedding
- Produces a vector fθ(x) for each MRI.
- Distances in this embedding represent relative quality differences (motion severity).

3.4.2 Adaptive Thresholding Mechanism
- Describe how the user sets a threshold or interacts with the system.
- Possibly define a continuous score function derived from the embedding.
- Mention potential adjustments per sequence type, if relevant.

3.4.3 Pivot-Based Visualization & Interactive UI
- Outline the interface concept: 
  - The user selects or adjusts desired quality levels,
  - Retrieves reference (pivot) images at different motion severity levels,
  - And applies pass/fail based on the chosen threshold.

--------------------------------------------------
4. EXPERIMENTS AND RESULTS
--------------------------------------------------
4.1 Datasets
4.1.1 Synthetic MRI Dataset
- Detailing how motion artifact is parameterized (e.g., random amplitude, phase errors, motion trajectories).
- Number of samples, FR-IQM distributions.
- **List the variety of sequences** simulated (T1, T2, etc.).

4.1.2 Real Clinical MRI Dataset
- Sample size, labeling process, inter-rater reliability (if applicable).
- **Indicate if multiple sequence types are included** (T1, T2, FLAIR, etc.), all focusing on motion artifact presence.

4.2 Implementation Details
- Model architecture specifics, hyperparameters, software libraries.
- Training procedure: epoch counts, learning rate scheduling, etc.

4.3 Quantitative Evaluation
4.3.1 Prediction of the FR-IQM
- Evaluate on a hold-out synthetic set: how well the model replicates the chosen FR-IQM for motion artifact severity.
- **Break down performance by sequence type** if relevant.

4.3.2 Expert Label Alignment
- Correlation (Spearman/Pearson), RMSE, or classification accuracy on the real dataset.
- **Sub-analysis per sequence** to see how well the model generalizes to motion in T1 vs. T2, etc.

4.3.3 Ablation Studies
- Without pairwise fine-tuning vs. with pairwise fine-tuning.
- Embedding-based scoring vs. direct regression baseline.
- Single-sequence vs. multi-sequence comparison, if data allows.

4.4 Qualitative Analysis
4.4.1 Embedding Visualization
- t-SNE or UMAP plots to show grouping by motion severity or sequence type.

4.4.2 Case Studies
- Compare how the system handles mild vs. severe motion scans in different sequences.
- Demonstrate pivot-based interface: how a user adapts thresholds and retrieves relevant images.

--------------------------------------------------
5. DISCUSSION
--------------------------------------------------
5.1 Strengths
- Versatility of embedding-based approach in capturing **motion artifact** across multiple sequences.
- Interactive potential for clinical workflows where a single threshold is insufficient.
- Potential to unify IQA across multiple MRI protocols.

5.2 Limitations
- Domain gap between synthetic (motion simulation) and real data; generalization challenges.
- Small real dataset might limit model capacity or require more domain adaptation.
- Need for more in-depth study of other confounding artifacts or large-scale real motion data.

5.3 Future Work
- Integration with multi-slice or 3D volumetric data.
- Extending the interactive interface to more MRI sequences or advanced motion simulation models.
- Exploring more advanced ranking losses or domain adaptation strategies for multi-sequence motion scenarios.

--------------------------------------------------
6. CONCLUSION
--------------------------------------------------
- Restate main outcomes and how this embedding-based method advances MRI IQA specifically for motion artifacts.
- Emphasize potential impact in real-world diagnostic settings.
- Note the viability of interactive retrieval for adaptively setting quality thresholds across various sequences; invite further improvements.

--------------------------------------------------
REFERENCES
--------------------------------------------------
[1] ...
[2] ...
[3] ...
(Use the citation format required by your target journal or conference.)
