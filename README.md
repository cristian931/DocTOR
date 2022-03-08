# DocTOR
This Repository represents the effort to obtain a predictor for single 
proteins associated to Adverse Reactions (ADRs).

DocTOR (Direct fOreCast Target On Reaction), is a utility written in
python3.9 (using the conda workframe) that allows the user to upload
a list of Uniprot IDs and Adverse reactions (from the available models) in order to study the 
relationship between the two. 

On output the program will assign a positive or negative class to 
the protein, assessing its possible involvement in the selected ADRs
onset.

DocTOR exploits the data coming from T-ARDIS [<https://doi.org/10.1093/database/baab068>] 
to train different Machine Learning approaches (SVM, RF, NN) using network 
topological measurements as features. 

The prediction coming from the single trained models are combined in a meta-predictor exploiting three different voting 
systems.  
* ___Jury Voting System___: One of the basic voting system, the output class will be simply the one with the highest amount of predictions.   
(e.g. Given a protein X, if two ML methods predict the class as 1 - or “ADR linked” and just one as 0, the ensemble method output class will be 1)


* ___Consensus Voting System___: This approach will sum the single ML class probability multiplied by 1 or -1 based if the prediction is respectively an ADR-linked or not. If the final sum of these values is positive the ensemble method output will be 1 or “ADR-linked”, if negative, 0 or “Not-ADR-linked”.   
(e.g. Given a protein X, if the SVM model predicts as class 1 with a probability of 0.86, RF predicts class 0 with a probability of 0.63 and the NN predicts as class 1 with a probability of 0.53, the overall class prediction will be equal to (0.86 * +1) + (0.63 * -1) + (0.53 * +1) = 0.76 ➡ class 1).


* ___Red Flag Voting System___: As opposed to the Jury Vote system, the Red Flag approach will select as output for the ensemble method the one with least predictions.  
(e.g. Given a protein X, if two ML methods predict the class as 1 - or “ADR linked” and just one as 0, the ensemble method output class will be 0)



The results of the meta-predictor together with the ones from the single ML method will be available in the output log 
file (named "predictions_community" or "predictions_curated" based on the database type). 

## Usage

DocTOR has been developed in a conda environment, to avoid problems and for the best usage is suggested to use the yml 
 file provided

1. Create the environment from the environment.yml file:  
`conda env create -f environment.yml`


2. Activate the new environment:  
`conda activate DocTOR_env`  


3. Download the trained models from <https://drive.google.com/drive/folders/1ge_ysgr_4kCBueTQZnCaIkwCfwhfSTIC?usp=sharing>


4. Extract the various tar.gz files (The entire uncompressed folder is ~20GB)  
`tar -zxf *.tar.gz`


5. Modify the ***adr_list*** and ***protein_list*** files with the Preferred Terms or SOC and Uniprot Ids of choice.  
The first accepts the adverse reaction reported in the ***Available_models*** file, the second accepts Uniprot Ids 
separated by newline.

    ***NB. It is suggested to run separately Preferred term and SOC search since the analysis uses different functions***  


6. Run the python script:  
In case of Preferred Terms selection:  

       python3 predictor_all_in_one.py protein_list adr_list PT   
    In case of System Organ Class selection:  

       python3 predictor_all_in_one.py protein_list adr_list SOC  
    If the ADR or Uniprot ID are not present in the database the program will raise a warning


7. The code will give as output two files based on the database type _predictions_community_ and _predictions_controlled_
both of them contains the prediction result in a tab delimited format.  
The file is divided in 15 columns:  
* ___BIANA ID___ : Protein code in the BIANA Network


* ___UNIPROT_ID___ : Uniprot ID of the protein in analysis


* ___adr___ : Adverse reaction in analysis


* ___top bench method___ : meta-predictor method that obtained the best results in terms of Accuracy Precision Recall and Matthew Correlation Coefficient during the independent test analysis for this Adverse Event. While also presenting the other methods, it is suggested to take this result as a possible reference


* ___jury_class___ : Attributed class based in the jury method (single ML method class result count) 
* ___vote_count___ : Number of votes for the class reported


* ___red_flag___ : Class reported by the Red Flag method


* ___consensus___ :Class reported by the Consensus method


* ___Predicted class SVM___ : Class predicted by the Support Vector Machine
* ___Predicted probability SVM___: Probability associated to the SVM class
* ___Predicted class RND___: Class predicted by the Random Forest method
* ___Predicted probability RND___: Probability associated to the RF class
* ___Predicted class NN___: Class predicted by the Neural Network method 
* ___Predicted probability NN___: Probability associated to the NN class


* ___Sum_Prob___ : Corrected probabilities for the Consensus Method 
