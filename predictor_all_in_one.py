import sys
import os
import pandas as pd
import keras
from joblib import load
from functools import reduce
from sklearn.preprocessing import StandardScaler
import numpy as np
import glob
import warnings

warnings.filterwarnings("ignore")  # remove sklearn warnings regarding loading an external model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove keras printed logs


def open_file(user_defined_protein_file, adr_file, biana_id_file):
    """
    Open and load user input file plus Biana id file
    :param user_defined_protein_file: (file) simple text file containing uniprot ID of the protein to search divided
    by newline
    :param adr_file: (file) simple text file containing the adverse event Preferred name (PT) based on the
    Meddra Classification
    :param biana_id_file: (file) Map file to convert the uniprot ids to Biana IDs
    :return df_protein_file: (dataframe) dataframe of proteins user defined
    :return user_defined_adr_list: (list) list of Adverse reaction name
    :return biana_dic_inverse: (dict) dictionary of biana-uniprot id entries
    """
    df_protein_file = pd.read_csv(user_defined_protein_file, sep='\t', names=['UNIPROT_ID'])
    df_biana = pd.read_csv(biana_id_file, sep='\t', names=['BIANA_ID', 'UNIPROT_ID'], dtype=str)
    biana_dic = dict(df_biana[['UNIPROT_ID', 'BIANA_ID']].values)
    biana_dic_inverse = dict(df_biana[['BIANA_ID', 'UNIPROT_ID']].values)
    df_protein_file['BIANA_ID'] = df_protein_file['UNIPROT_ID'].map(biana_dic)
    list_se = pd.read_csv(adr_file, sep='\t', names=['adr'])
    user_defined_adr_list = list_se['adr'].to_list()

    return df_protein_file, user_defined_adr_list, biana_dic_inverse


def load_and_prepare(input_dataframe, meddra_type, selection, df_protein_file):
    """
    Parse and prepare the training set for the prediction selecting just the protein that the used defined related to
    the side effect
    :param input_dataframe: Dataframe containing the numerical feature for the training sets, the files are the one with
    extension _outdf
    :param meddra_type: (str) type of meddra classification to search - PT (preferred term) / SOC (System Organ Class)
    :param selection: (str) name of the Adverse event in analysis (PT / SOC)
    :param df_protein_file: (dataframe) dataframe of protein to predict - user defined
    :return testing_Scaled_X: (dataframe) Normalized features of protein to predict
    :return feature_in_analysis: (dataframe) un-normalized dataframe of protein to predict
    """
    df = pd.read_csv(input_dataframe, sep='\t')

    df_in_analysis = df[df[meddra_type] == selection].reset_index(drop=True)

    for feature in ['BIANA ID',
                    'module ID K1',
                    'module ID LN'
                    ]:
        df_in_analysis[feature] = df_in_analysis[feature].astype(str)

    df_in_analysis = pd.get_dummies(
        df_in_analysis,
        columns=['module ID K1',
                 'module ID LN'
                 ]
    )
    if meddra_type == 'SOC':
        dataframe_columns_to_remove = ['BIANA ID',
                                       'Side_effect',
                                       'Seed',
                                       'SOC',
                                       'Diamond Score'
                                       ]
    else:
        dataframe_columns_to_remove = ['BIANA ID',
                                       'Side_effect',
                                       'Seed',
                                       'PT',
                                       'HLT',
                                       'HLGT',
                                       'SOC',
                                       'Diamond Score'
                                       ]

    feature_in_analysis = df_in_analysis[df_in_analysis['BIANA ID'].isin(df_protein_file['BIANA_ID'].to_list())]

    proteins_to_predict = feature_in_analysis.drop(dataframe_columns_to_remove, axis=1)

    scaler = StandardScaler()
    testing_scaled_X = scaler.fit_transform(proteins_to_predict)

    return testing_scaled_X, feature_in_analysis


def model_load(model_list, adr, database_type):
    """
    Load the best model from the independent tested models, report the best meta-predictor method based on the
    independent testing
    :param model_list: (dataframe) dataframe of best obtained models with relative score and positive/negative ratio
    :param adr: (str) name of the Adverse reaction
    :param database_type: (str) TARDIS origin database data, community/controlled
    :return model_SVM: (skleran model) Load the best SVM model for the current ADR
    :return model_RAND: (skleran model) Load the best RF model for the current ADR
    :return model_NN: (keras model) Load the best NN model for the current ADR
    :return best_method: (str) Report the
    """

    model_list['frac'] = model_list['frac'].apply(lambda x: x.replace('(', '').replace(')', '').split(', '))
    model_frac = model_list[model_list['adr'] == adr]['frac'].to_list()[0]

    MCCs = model_list[model_list['adr'] == adr][['jury_MCC', 'red_MCC', 'consensus_MCC']]
    best_method = MCCs.idxmax(axis=1).tolist()[0]
    if best_method == 'jury_MCC':
        best_method = 'Jury Method'
    elif best_method == 'red_MCC':
        best_method = 'Red Flag Method'
    elif best_method == 'consensus_MCC':
        best_method = 'Consensus Method'

    model_name_sklearn = 'model_' + adr + '_' + model_frac[0] + '_neg_frac_' + model_frac[1] + '.joblib'
    model_name_keras_architecture = adr + '_' + model_frac[0] + '_neg_frac_' + model_frac[1] + '_.h5'

    model_file_SVM = glob.glob('SVM_models_' + database_type + '/' + model_name_sklearn)
    model_file_RAND = glob.glob('RAND_models_' + database_type + '/' + model_name_sklearn)
    model_file_NN_architecture = glob.glob('NN_models_' + database_type + '_resaved/' + model_name_keras_architecture)

    model_SVM = load(model_file_SVM[0])
    model_RAND = load(model_file_RAND[0])

    print(model_file_NN_architecture)

    # NN_json_file = open(model_file_NN_architecture[0], 'r')
    # loaded_model_json = NN_json_file.read()
    # NN_json_file.close()
    model_NN = keras.models.load_model(model_file_NN_architecture[0], compile=False)
    # load weights into new model
    # model_NN.load_weights(model_file_NN_weights[0])
    model_NN.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

    return model_SVM, model_RAND, model_NN, best_method


def predictions_sklearn(model, scaled_feature_file, all_feature_file, ml_type):
    """
    :param model: (Sklearn model) Selected model (SVM, RF)
    :param scaled_feature_file: (dataframe) Normalized features of selected proteins
    :param all_feature_file: (dataframe) Original feature file to retrieve the Associated Biana ID
    :param ml_type: (str) Machine learning label (sklearn/keras)
    :return pred_df_1: (dataframe) Dataframe containing the prediction result and relative probabilities
    """
    ypred_bst = model.predict(scaled_feature_file)
    ypred_proba = model.predict_proba(scaled_feature_file)

    pred_df_1 = pd.DataFrame(
        list(zip(all_feature_file['BIANA ID'].to_list(), ypred_bst, ypred_proba)),
        columns=['BIANA ID', 'Predicted class ' + ml_type, 'Predicted probability ' + ml_type]
    )
    pred_df_1['Predicted probability ' + ml_type] = pred_df_1['Predicted probability ' + ml_type].apply(
        lambda x: max(x)
        )

    return pred_df_1


def predictions_keras(model, scaled_feature_file, all_feature_file, ml_type):
    """
    :param model: (keras model) Selected model (NN)
    :param scaled_feature_file: (dataframe) Normalized features of selected proteins
    :param all_feature_file: (dataframe) Original feature file to retrieve the Associated Biana ID
    :param ml_type: Machine learning label (sklearn/keras)
    :return pred_df_1: (dataframe) Dataframe containing the prediction result and relative probabilities
    """
    ypred_bst = model.predict(scaled_feature_file)
    ypred_bst_class = np.where(ypred_bst > 0.5, 1, 0)
    pred_df_1 = pd.DataFrame(
        list(
            zip(
                all_feature_file['BIANA ID'].to_list(),
                ypred_bst_class,
                ypred_bst
            )
        ),
        columns=['BIANA ID', 'Predicted class ' + ml_type, 'Predicted probability ' + ml_type]
    )

    pred_df_1['real_prob'] = pred_df_1.apply(
        lambda x: (1 - x['Predicted probability ' + ml_type]) if x['Predicted class ' + ml_type] == 0 else x[
            'Predicted probability ' + ml_type], axis=1
    )

    pred_df_1['Predicted class ' + ml_type] = pred_df_1['Predicted class ' + ml_type].apply(lambda x: list(x)[0])
    pred_df_1['Predicted probability ' + ml_type] = pred_df_1['real_prob'].apply(lambda x: list(x)[0])
    pred_df_1.drop(columns=['real_prob'], inplace=True)

    return pred_df_1


def dataframe_prep_batch(prob_svm, prob_rnd, prob_nn, biana_dic):
    """
    Merge and clean the prediction databases created
    :param prob_svm: (dataframe) SVM prediction dataframe containing predicted class and probabilities
    :param prob_rnd: (dataframe) RF prediction dataframe containing predicted class and probabilities
    :param prob_nn: (dataframe) NN prediction dataframe containing predicted class and probabilities
    :param biana_dic: (dict) dictionary to map between Uniprot IDs and Biana IDs
    :return df2: (dataframe) Merged dataframe with removed redundant columns
    """
    data_frames = [prob_svm, prob_rnd, prob_nn]
    # merge sequentially all dataframes using the BIANA ID as key
    df_merg = reduce(lambda left, right: pd.merge(left, right, on='BIANA ID', how='inner'), data_frames)
    df1 = df_merg[df_merg.columns.drop(list(df_merg.filter(regex='_x')))]  # drop duplicate columns (same name added _x)
    df2 = df1[df1.columns.drop(list(df1.filter(regex='_y')))]  # drop duplicate columns (same name added _y)
    df2['UNIPROT_ID'] = df2['BIANA ID'].map(biana_dic)

    return df2


def sum_probability(df):
    """
    Definition of probability function for the meta-predictor
    [if class prediction == 0 compute -1 * prob
    elif class prediction == 1 compute 1 * prob

    at the end of the procedure the resulting meta-predictor score is the sum of the weighted probabilities:
    if <= 1 => class prediction 1
    else >= 1 => class prediction 0]
    :param df: (dataframe) Merged prediction database
    :return prob_sum: (float) sum of weighted probabilities
    """
    if df['Predicted class SVM'] == 0 or df['Predicted class SVM'] == '0':
        prob_adjusted_SVM = -1*float(df['Predicted probability SVM'])
    else:
        prob_adjusted_SVM = 1*float(df['Predicted probability SVM'])

    if df['Predicted class RND'] == 0 or df['Predicted class RND'] == '0':
        prob_adjusted_RND = -1*float(df['Predicted probability RND'])
    else:
        prob_adjusted_RND = 1*float(df['Predicted probability RND'])

    if df['Predicted class NN'] == 0 or df['Predicted class NN'] == '0':
        prob_adjusted_NN = -1*float(df['Predicted probability NN'])
    else:
        prob_adjusted_NN = 1*float(df['Predicted probability NN'])

    prob_sum = prob_adjusted_SVM + prob_adjusted_RND + prob_adjusted_NN

    return prob_sum


def voting_systems(df):
    """
    Function for computing the met-predictor scores
    :param df: (dataframe) Merged prediction database
    :return df: (dataframe) Merged prediction database with final meta-predictor computed class and probabilities
    """
    df['Sum_Prob'] = df.apply(lambda x: sum_probability(x), axis=1)
    df['jury_class'] = df['Sum_Prob'].apply(lambda x: 1 if (x > 0) else 0)

    df['vote_count'] = df.apply(
        lambda x: int(x['Predicted class SVM']) + int(x['Predicted class RND']) + int(x['Predicted class NN']),
        axis=1
    )

    df['red_flag'] = df['vote_count'].apply(lambda x: 0 if (x == 0 or x == 2) else 1)

    df['consensus'] = df['vote_count'].apply(lambda x: 1 if x >= 2 else 0)

    return df


if __name__ == '__main__':
    protein_file_input = sys.argv[1]
    adr_file_input = sys.argv[2]
    meddra_group = sys.argv[3]

    community = 'community'
    controlled = 'controlled'

    protein_file, adr_list, biana_dictionary = open_file(
        protein_file_input,
        adr_file_input,
        biana_id_file='node_info_BIANA.txt'
        )

    for db_type in [community, controlled]:
        top_model_df = pd.read_csv('best_models_' + db_type, sep='\t')
        empty_df = pd.DataFrame()
        for se in adr_list:
            if se in top_model_df['adr'].to_list():
                svm_, rand_, NN_, top_method = model_load(top_model_df, se, db_type)

                if meddra_group == 'SOC':
                    scaled_feature, all_features = \
                        load_and_prepare(db_type + '_SOC_outdf', meddra_group, se, protein_file)
                else:
                    scaled_feature, all_features = \
                        load_and_prepare(db_type + '_outdf', meddra_group, se, protein_file)

                svm_pred = predictions_sklearn(svm_, scaled_feature, all_features, 'SVM')
                rand_pred = predictions_sklearn(rand_, scaled_feature, all_features, 'RND')
                NN_pred = predictions_keras(NN_, scaled_feature, all_features, 'NN')

                all_predictions = dataframe_prep_batch(svm_pred, rand_pred, NN_pred, biana_dictionary)

                final_df = voting_systems(all_predictions)

                final_df['adr'] = se

                final_df['top bench method'] = top_method

                empty_df = pd.concat([empty_df, final_df])
            else:
                if db_type == 'controlled':
                    print('The selected ADR ' + se + ' is not in the curated database')
                    continue
                else:
                    print('The selected ADR ' + se + ' is not in the community database')
                    continue

        out_df = empty_df[['BIANA ID',
                           'UNIPROT_ID',
                           'adr',
                           'top bench method',
                           'jury_class',
                           'vote_count',
                           'red_flag',
                           'consensus',
                           'Predicted class SVM',
                           'Predicted probability SVM',
                           'Predicted class RND',
                           'Predicted probability RND',
                           'Predicted class NN',
                           'Predicted probability NN',
                           'Sum_Prob']]

        out_df.to_csv('predictions_' + db_type, sep='\t', index=False)
