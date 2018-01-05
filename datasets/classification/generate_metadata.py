import csv

from pmlb.dataset_lists import classification_dataset_names
from pmlb import fetch_data
from pmlb.write_metadata import imbalance_metrics, count_features_type


def compute_class_summary(target):
    class_counts = target.value_counts()
    majority_size, minority_size = class_counts.max(), class_counts.min()
    num_classes, imbalance = imbalance_metrics(target.tolist())
    return majority_size, minority_size, num_classes, imbalance


def compute_missingness_summary(df):
    instances_without_missing = df.dropna(axis=0).shape[0]
    instances_with_missing_values = df.shape[0] - instances_without_missing
    total_missing_values = df.isnull().sum().sum()
    return instances_with_missing_values, total_missing_values


def get_classification_dataset_summary(df, dataset):
    # Class summary
    majority_size, minority_size, num_classes, imbalance = compute_class_summary(df["target"])

    # Feature summary
    num_instances, num_features = df.drop('target', axis=1).shape
    binary_features, integer_features, float_features = count_features_type(df.drop('target', axis=1))
    numeric_features = integer_features + float_features
    symbolic_features = num_features - numeric_features

    # Missing data summary
    instances_with_missing_values, total_missing_values = compute_missingness_summary(df)

    summary = {
        "MajorityClassSize": majority_size,
        "MinorityClassSize": minority_size,
        "NumberOfClasses": num_classes,
        "ImbalanceMetric": imbalance,
        "NumberOfFeatures": num_features,
        "NumberOfBinaryFeatures": binary_features,
        "NumberOfIntegerFeatures": integer_features,
        "NumberOfFloatFeatures": float_features,
        "NumberOfInstances": num_instances,
        "NumberOfInstancesWithMissingValues": instances_with_missing_values,
        "NumberOfMissingValues": total_missing_values,
        "NumberOfNumericFeatures": numeric_features,
        "NumberOfSymbolicFeatures": symbolic_features,
        "name": dataset,
        "status": "active"
    }

    return summary


if __name__ =='__main__':
    local_dir = '.'

    tsv_fieldnames = [
        "MajorityClassSize",
        "MinorityClassSize",
        "NumberOfClasses",
        "ImbalanceMetric",
        "NumberOfFeatures",
        "NumberOfBinaryFeatures",
        "NumberOfIntegerFeatures",
        "NumberOfFloatFeatures",
        "NumberOfInstances",
        "NumberOfInstancesWithMissingValues",
        "NumberOfMissingValues",
        "NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures",
        "name",
        "status"
    ]

    with open('classification_datasets_pmlb.tsv', 'w') as csvfile:
        try:
            writer = csv.DictWriter(csvfile, fieldnames=tsv_fieldnames, delimiter='\t')
            writer.writeheader()

            for dataset in classification_dataset_names:
                print(dataset, '...')
                df = fetch_data(dataset)
                summary_dict = get_classification_dataset_summary(df, dataset)

                writer.writerow(summary_dict)
        except csv.Error as err:
            print(err)
