import numpy as np
import csv
import datetime
import json
import os
from glob import glob

if not os.path.exists('Preprocessed_Data'):
    os.mkdir('Preprocessed_Data')

csvs = glob('Data/*.csv')
for file_path in csvs:
    if not 'submit' in file_path:
        labels = []
        X = {}
        Y = {}
        base_date = None
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for i,row in enumerate(reader):
                if i == 0:
                    for label in row:
                        if label != '':
                            labels.append(label)
                            if 'Damage' in label:
                                Y[label] = np.zeros([1008])-1
                            else:
                                X[label] = np.zeros([1008])-1

                elif i != 1:
                    for j,label in enumerate(labels):
                        if row[j*3] != '':

                            time_stamp = datetime.datetime.strptime(row[j*3].split(' ')[-1], '%H:%M')
                            minute_index = round(time_stamp.minute/10)
                            hour_index = time_stamp.hour
                            if base_date == None:
                                base_date = datetime.datetime.strptime(row[j*3].split(' ')[0], '%m/%d/%y')

                            day = (datetime.datetime.strptime(row[j*3].split(' ')[0], '%m/%d/%y')-base_date).days

                            time_index = day * 24 * 6 - 31 + minute_index + hour_index * 6

                            if 'Damage' in label:
                                Y[label][time_index] = float(row[j*3 + 1])
                            else:
                                X[label][time_index] = float(row[j*3 + 1])
            X_headers = []
            Y_headers = []
            for i,d in enumerate(X):
                X_headers.append(d)
                if i == 0:
                    X_np = np.expand_dims(X[d],-1)
                else:
                    X_np = np.concatenate([X_np,np.expand_dims(X[d],-1)],-1)

            for i,d in enumerate(Y):
                Y_headers.append(d)
                if i == 0:
                    Y_np = np.expand_dims(Y[d],-1)
                else:
                    Y_np = np.concatenate([Y_np,np.expand_dims(Y[d],-1)],-1)
            file_path = file_path.replace('Data','Preprocessed_Data')

            np.save(file_path.replace('.csv','-X.npy'),X_np)
            
            if not 'test' in file_path:
                np.save(file_path.replace('.csv','-Y.npy'),Y_np)

            with open(file_path.replace('.csv','-X-Headers.json'), 'w') as outfile:
                json.dump(X_headers, outfile)
            if not 'test' in file_path:
                with open(file_path.replace('.csv','-Y-Headers.json'), 'w') as outfile:
                    json.dump(Y_headers, outfile)