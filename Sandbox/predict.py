# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

#pip install keras
#pip install tensorflow

import argparse
import os
import pandas as pd
import numpy as np
import time

#from covid_xprize.examples.predictors.lstm.xprize_predictor import XPrizePredictor
from xprize_predictor import XPrizePredictor
from HaloTransformFinal import HaloTransform

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# If you'd like to use a model, copy it to "trained_model_weights.h5"
# or change this MODEL_FILE path to point to your model.
MODEL_WEIGHTS_FILE = os.path.join(ROOT_DIR, "models", "trained_model_weights.h5")

DATA_FILE = os.path.join(ROOT_DIR, 'data', "review_model_data.csv")


def predict(start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path) -> None:
    """Generates and saves a file with daily new cases predictions for the
    given countries, regions and intervention plans, between start_date and
    end_date, included.

    :param start_date: day from which to start making predictions, as a
    string, format YYYY-MM-DDD :param end_date: day on which to stop
    making predictions, as a string, format YYYY-MM-DDD :param
    path_to_ips_file: path to a csv file containing the intervention
    plans between inception date (Jan 1 2020)  and end_date, for the
    countries and regions for which a prediction is needed :param
    output_file_path: path to file to save the predictions to :return:
    Nothing. Saves the generated predictions to an output_file_path CSV
    file with columns
    "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # !!! YOUR CODE HERE !!!
    predictor = XPrizePredictor(MODEL_WEIGHTS_FILE, DATA_FILE)
    # Generate the predictions
    preds_df = predictor.predict(start_date, end_date, path_to_ips_file)
    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    #NEW CODE
    df = preds_df
    inDates = df['Date']
    inDate = []
    for x in inDates:
        #print(x)
        #y = x.replace("-","")
        stime = str(x)
        y = '%s%s%s'%(stime[0:4],stime[5:7],stime[8:10])
        inDate.append(y)
    inCountry = df['CountryName']
    inRegion = df['RegionName']
    inRegion = inRegion.replace(np.NaN,"all")
    inCovid = df['PredictedDailyNewCases']
    outCovid,outSp,outWord = HaloTransform(inDate,inCountry,inRegion,inCovid)

    df.rename(columns={'PredictedDailyNewCases': 'DailyNewCases_Pre_HALO'}, inplace=True)
    df.insert(loc=3, column='PredictedDailyNewCases', value=outCovid)
    df.insert(loc=5, column='IsSpecialty', value=outSp)
    df.insert(loc=6, column='Words', value=outWord)
    

    # print("df covid2 dataframe",df['PredictedDailyNewCases2'])
    # Save to a csv file
    df.to_csv(output_file_path, index=False)
    print(f"Saved predictions to {output_file_path}")
          
    #write file with top words:
    # outfile = open('work/predictions/MostCommonWords_%d.txt'%int(time.time()),'w')
    # CtWords = []
    # for a in range (0,len(outWord)):
    #     CtWords.append(outWord.count(outWord[a]))
    # sCMW=[x for _,x in sorted (zip(CtWords,outWord),reverse=True)]
    # if 100 >= len(sCMW):
    #     LenOutfile = len(sCMW)
    # else:
    #     LenOutfile = 100
    # for a in range (0,LenOutfile-1):
    #     outfile.write('%s,'%sCMW[a])
    # outfile.write('%s'%sCMW[-1])
    # outfile.close()
    

# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to the CSV file where predictions should be written")
    args = parser.parse_args()
    print(f"Generating predictions from {args.start_date} to {args.end_date}...")
    predict(args.start_date, args.end_date, args.ip_file, args.output_file)
    print("Done!")