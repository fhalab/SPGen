Repository for analysis of experimental results obtained testing the machine-generated SPs in Bacillus subtilis.

The appropriate environment may be found in `validation_env.yml`.

Initial data obtained from screening can be found in data/191203_mastersheet_BGsubtracted_raw.xls

### Section 1
This data is preprocessed and tidied with 1_basf_preprocessing.ipynb, to check for presence correctly cloned Signal Peptides and Enzymes. 
    - The processed data is output in data/preprocessed.csv
    
### Section 2 contains notebooks for processing enzymatic assay data. 

2a_negative_control_analysis.ipynb is then used to classify constructs as functional or nonfunctional based, based on statistical significant and effect size. This notebook is used to generate the Figures in Supplemental Section 2

2b_summary_stats.ipynb contains summary statistics generated from 2a_..., which can be found in the Supplemental Section 2.

2c_negative_control_analysis_higherdilution.ipynb repeats notebook 2a for the xylanase and amylase enzymatic families, as the best-performing constructs

2d_top_func_figures.ipynb generates the figures found in Figure 2, based on data from 2a or 2c as appropriate (within the linear range of the assays).

### Section 3 contains the SignalP comparison.

3_func_signalp.ipynb generates an ROC curve with SignalP 5.0 predictions.
3a_signalP_comparison_generation.ipynb contains sample functions for formatting required sequences for the SignalP 5.0 server.

### Section 4

4_func_v_non_comparison.ipynb contains basic comparisons between functional and nonfunctional generated SPs, and is used to generate Supplemental Section 4.

### Section 5

5_alignment_remote/ contains files and folders for generating MSAs found in Supplemental Section 5, which is essentially entirely based on Damian Ferrell's repo: https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner. Note that MUSCLE must be installed for the alignment generation.

5_alignment_remote/5a_SP_blast.ipynb performs pairwise alignments between all generated and natural SPs to identify % sequence identity. 
5_alignment_remote/5b_msa_generation.py generates MSA figures, with small tweaks of Damin Ferrell's work.
5_alignment_remote/5c_seq_distr_swarm.ipynb generates the sequence identity figure in Figure 3a: Percent sequence identity of tested machine generated SPs to the closest matching natural SPs in UniProt.

### Section 6

Section 6 contains information needed to recreate Supplemental Section 1, which compared a profile Hidden Markov Model, heuristic, and VAE -generated sequences. 
For the pHMM, S. Eddy's HMMER software was used. HMMER requires an alignment. After experimenting with hmmalign and Clustal, we ultimately decided to use an alignment from Clustal, which is limited to 4000 sequences. To this end, we randomly selected 4000 sequences to generate an alignment with default settings.

For predicted signal peptide probabilities, Nielsen's SignalP server was used.
