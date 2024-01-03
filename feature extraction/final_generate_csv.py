import os
import csv
import random

feature_path = './dataset/1128/feature'

out_test_csv = './test.csv'


if __name__ == '__main__':
    # random.seed(230907)
    features = os.listdir(feature_path)
    features = [v for v in features]
    with open(out_test_csv, 'w') as f_test:
        f_test_csv = csv.writer(f_test, delimiter=',')
        for i, features_name in enumerate(features):
            features_path = 'dataset/1128/feature/{}'.format(features_name)
            instance_count_path = 'out_1128/features/instance_count/{}'.format(features_name)
            instance_name_path = 'out_1128/features/instance_name/{}z'.format(features_name[:-1])
            f_test_csv.writerow([features_path, instance_count_path, instance_name_path])