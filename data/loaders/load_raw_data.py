import pyarrow.parquet as pq
from datasets import load_dataset
import os
import pandas as pd
from typing import Tuple, Dict


def dataset_to_dataframe(dataset_dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a DatasetDict to a tuple of pandas DataFrames (train_df, test_df).
    
    Args:
        dataset_dict: A DatasetDict object with 'train' and 'test' splits.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    train_df = dataset_dict["train"].to_pandas()
    test_df = dataset_dict["test"].to_pandas()
    
    return train_df, test_df


def build_label_mappings(
    dataset_dict,
    label_field: str = "label",
    text_label_field: str = "text_label",
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create bidirectional mappings between numeric labels and text labels.

    Args:
        dataset_dict: A DatasetDict containing (at least) a "train" split with
            both numeric label and text label columns.
        label_field: Name of the numeric label column.
        text_label_field: Name of the textual label column.

    Returns:
        text_label_to_label: {text_label -> label_id}
        label_to_text_label: {label_id -> text_label}
    """

    if "train" not in dataset_dict:
        raise ValueError("DatasetDict must contain a 'train' split to build mappings.")

    train_split = dataset_dict["train"]
    if label_field not in train_split.column_names:
        raise ValueError(f"Column '{label_field}' not found in train split.")
    if text_label_field not in train_split.column_names:
        raise ValueError(f"Column '{text_label_field}' not found in train split.")

    labels = train_split[label_field]
    text_labels = train_split[text_label_field]

    label_to_text_label: Dict[int, str] = {}
    text_label_to_label: Dict[str, int] = {}

    for lbl, txt in zip(labels, text_labels):
        lbl_int = int(lbl)
        # Prefer first occurrence to avoid inconsistencies
        label_to_text_label.setdefault(lbl_int, txt)
        text_label_to_label.setdefault(txt, lbl_int)

    return text_label_to_label, label_to_text_label


def load_raw_hwu64(
        data_dir:str="/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/data/raw/hwu64_deeppavlov",
        return_dicts:bool=False
    ):
        raw_dataset=load_dataset(
                "parquet",
                data_files={
                        "train": os.path.join(data_dir, "data","train-00000-of-00001.parquet"),
                        "test": os.path.join(data_dir, "data","test-00000-of-00001.parquet"),
                }
        )
        label_map=load_dataset(
                "parquet",
                data_files={
                        "intents": os.path.join(data_dir, "intents","intents-00000-of-00001.parquet"),
                },
        )
        label_list=label_map["intents"]["name"]
        
        # Create a mapping from label id to label name
        label_id_to_name = {i: name for i, name in enumerate(label_list)}
        
        # Add text_label field to each split
        def add_text_label(example):
            example["text"]=example["utterance"]
            del example["utterance"]
            example["text_label"] = label_id_to_name[example["label"]].replace("_"," ")
            return example
        
        raw_dataset = raw_dataset.map(add_text_label)
        
        # Calculate dataset statistics
        dataset_text_num = {
            "train": len(raw_dataset["train"]),
            "test": len(raw_dataset["test"]),
            "num_labels": len(label_list),
            "label_names": label_list
        }
        if return_dicts:
            text_label2label, label2text_label = build_label_mappings(raw_dataset)
            return raw_dataset, dataset_text_num, text_label2label, label2text_label
        return raw_dataset, dataset_text_num

def load_raw_atis(
          data_dir:str="/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/data/raw/atis_intents_fathyshalab",
          return_dicts:bool=False
    ):
        raw_dataset=load_dataset(
                "parquet",
                data_files={
                        "train": os.path.join(data_dir, "data","train-00000-of-00001-c33055e2fcb4738f.parquet"),
                        "test": os.path.join(data_dir, "data","test-00000-of-00001-63d456c01095f1b2.parquet"),
                }
        )
        def change_label_name(example):
            example["text_label"]=example["label text"].replace("atis_","").replace("_"," ")
            del example["label text"]
            return example
        raw_dataset=raw_dataset.map(change_label_name)
        # Calculate dataset statistics
        label_list=raw_dataset["train"].unique("text_label")
        dataset_text_num = {
            "train": len(raw_dataset["train"]),
            "test": len(raw_dataset["test"]),
            "num_labels": len(label_list),
            "label_names": label_list
        }
        if return_dicts:
            text_label2label, label2text_label = build_label_mappings(raw_dataset)
            return raw_dataset, dataset_text_num, text_label2label, label2text_label
        return raw_dataset, dataset_text_num

def load_raw_snips(
        data_dir:str="/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/data/raw/snips_benayas",
        return_dicts:bool=False
    ):
        raw_dataset=load_dataset(
                "parquet",
                data_files={
                        "train": os.path.join(data_dir, "data","train-00000-of-00001.parquet"),
                        "test": os.path.join(data_dir, "data","test-00000-of-00001.parquet"),
                }
        )
        
        # Rename 'category' to 'text_label'
        def rename_category(example):
            example["text_label"] = example["category"]
            del example["category"]
            return example
        
        raw_dataset = raw_dataset.map(rename_category)
        
        # Get unique label names from train set
        label_list = sorted(raw_dataset["train"].unique("text_label"))
        
        # Create mapping from text_label to numeric label
        label_name_to_id = {name: i for i, name in enumerate(label_list)}
        
        # Add numeric 'label' field
        def add_numeric_label(example):
            example["label"] = label_name_to_id[example["text_label"]]
            return example
        
        raw_dataset = raw_dataset.map(add_numeric_label)
        
        # Calculate dataset statistics
        dataset_text_num = {
            "train": len(raw_dataset["train"]),
            "test": len(raw_dataset["test"]),
            "num_labels": len(label_list),
            "label_names": label_list
        }

        if return_dicts:
            text_label2label, label2text_label = build_label_mappings(raw_dataset)
            return raw_dataset, dataset_text_num, text_label2label, label2text_label
        return raw_dataset, dataset_text_num


def load_raw_clinc_oos(
        data_dir:str="/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/data/raw/clinc_oos_deeppavlov",
        version: str="plus",
        return_dicts:bool=False
    ):
        valid_versions=["small","plus","imbalanced"]
        if version not in valid_versions:
            raise ValueError(f"Invalid version: {version}. Must be one of {valid_versions}.")
        raw_dataset=load_dataset(
                "parquet",
                data_files={
                        "train": os.path.join(data_dir, version,"train-00000-of-00001.parquet"),
                        "test": os.path.join(data_dir, version,"test-00000-of-00001.parquet"),
                }
        )
        def change_label_name(example):
            example["text_label"]=example["label_text"].replace("_"," ")
            if example["text_label"]=="oos":
                example["text_label"]="out of scope"
            del example["label_text"]
            return example
        raw_dataset=raw_dataset.map(change_label_name)
        # Calculate dataset statistics
        label_list=raw_dataset["train"].unique("text_label")
        dataset_text_num = {
            "train": len(raw_dataset["train"]),
            "test": len(raw_dataset["test"]),
            "num_labels": len(label_list),
            "label_names": label_list
        }
        if return_dicts:
            text_label2label, label2text_label = build_label_mappings(raw_dataset)
            return raw_dataset, dataset_text_num, text_label2label, label2text_label
        return raw_dataset, dataset_text_num



def load_raw_project(
        data_dir:str="/usr1/home/s125mdg41_03/code/Intent_Recognition_Exp/data/raw/project", 
        return_dicts:bool=False
):
        raw_dataset=load_dataset(
                "json",
                data_files={
                        "train": os.path.join(data_dir,"train.json"),
                        "test": os.path.join(data_dir,"test.json"),
                }
        )
        
        # Process train split: has sentiments and reviews columns
        def process_train(example):
            example["label"]=int(example["sentiments"])
            del example["sentiments"]
            example["text"]=example["reviews"]
            del example["reviews"]
            example["text_label"]="positive" if example["label"]==1 else "negative"
            return example
        
        # Process test split: only has reviews column (rename to text)
        def process_test(example):
            example["text"]=example["reviews"]
            del example["reviews"]
            return example
        
        # Map separately for train and test
        raw_dataset["train"] = raw_dataset["train"].map(process_train)
        raw_dataset["test"] = raw_dataset["test"].map(process_test)
        
        # Calculate dataset statistics
        label_list=raw_dataset["train"].unique("text_label")
        dataset_text_num = {
            "train": len(raw_dataset["train"]),
            "test": len(raw_dataset["test"]),
            "num_labels": len(label_list),
            "label_names": label_list
        }
        if return_dicts:
            return raw_dataset, dataset_text_num, {"negative":0,"positive":1},{0:"negative",1:"positive"}
        return raw_dataset, dataset_text_num



if __name__ == "__main__":
    # Print class distributions for each dataset (text_label + numeric)
    def print_dist(title: str, df: pd.DataFrame):
        print(f"\n{title}")
        if "text_label" in df.columns:
            counts = df["text_label"].value_counts().sort_values(ascending=False)
            print(f"- num_labels: {df['text_label'].nunique()}")
            # Show all categories in full
            print(counts.to_string(max_rows=None))
        else:
            print("- Column text_label missing; skip text label distribution")
        if "label" in df.columns:
            counts_num = df["label"].value_counts().sort_index()
            print("- numeric labels distribution (by id):")
            # Show numeric label counts
            print(counts_num.to_string(max_rows=None))
        else:
            print("- Column label missing; skip numeric distribution")

    def report_dataset(name: str, dataset_dict):
        print("\n" + "="*80)
        print(f"[Label Distribution] Dataset: {name}")
        print("="*80)
        train_df, test_df = dataset_to_dataframe(dataset_dict)
        print_dist("Train split:", train_df)
        print_dist("Test split:", test_df)

    from contextlib import redirect_stdout

    # Where to save the report
    report_dir = "/usr1/home/s125mdg41_03/code/Intent_Recognition_Exp"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "label_distributions.txt")

    # Redirect the following stdout to file
    with open(report_path, "w", encoding="utf-8") as f, redirect_stdout(f):
        # Avoid truncated pandas output
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.width", 200)
        print("="*80)
        print("All datasets label distributions")
        print("="*80)

        atis_ds, _, text_label2id, id2text_label = load_raw_atis(return_dicts=True)
        report_dataset("atis_intents_fathyshalab", atis_ds)
        print("    - text_label2id:", text_label2id)
        print("    - id2text_label:", id2text_label)
        hwu64_ds, _, text_label2id, id2text_label = load_raw_hwu64(return_dicts=True)
        report_dataset("hwu64_deeppavlov", hwu64_ds)
        print("    - text_label2id:", text_label2id)
        print("    - id2text_label:", id2text_label)
        snips_ds, _, text_label2id, id2text_label = load_raw_snips(return_dicts=True)
        report_dataset("snips_benayas", snips_ds)
        print("    - text_label2id:", text_label2id)
        print("    - id2text_label:", id2text_label)
        clinc_plus_ds, _, text_label2id, id2text_label = load_raw_clinc_oos(version="plus", return_dicts=True)
        report_dataset("clinc_oos_deeppavlov/plus", clinc_plus_ds)
        print("    - text_label2id:", text_label2id)
        print("    - id2text_label:", id2text_label)
    # Console notice for saved path
    print(f"Label distributions report saved to: {report_path}")
