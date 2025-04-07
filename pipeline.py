import pandas as pd
from src.category_tree.category_tree import CategoryTree
from sklearn.model_selection import GroupKFold, train_test_split

#########################

CAT_PATH = "data/raw/category_tree.csv"
TRAIN_LABELED_PATH = "data/raw/labeled_train.parquet"
UNLABELED_SPECIAL_PRIZE_PATH = "data/raw/unlabeled_special_prize.parquet"
UNLABELED_PATH = "data/raw/unlabeled_train.parquet"
UNLABELED_DORAZMETKA_PATH = 'data/raw/unlabeled_dorazmetka.parquet'
NEW_ITEMS_PATH = 'data/raw/new_items.parquet'

TRAIN_DATASET_PATH = "data/processed/train.parquet"
VAL_DATASET_PATH = 'data/processed/val.parquet'
BAD_LABELED_PATH = 'data/processed/bad_labeled.parquet'

CAT_ID_COL = "cat_id"
TITLE_COL = "source_name"
HASH_ID_COL = 'hash_id'
ATTR_COL = 'attributes'
PART_TYPE_COL = "part_type"
PART_COL = "part"

RANDOM_STATE = 42
TEST_PART_SIZE = 0.1

category_tree = CategoryTree(category_tree_path=CAT_PATH)

########################

# Step 1: Split labeled_train into good_labeled, bad_labeled, and hold out test set**
# good_labeled: samples with leaf category labels, bad_labeled: samples with non-leaf category labels

labeled_train = pd.read_parquet(TRAIN_LABELED_PATH, columns=[TITLE_COL, CAT_ID_COL, HASH_ID_COL, ATTR_COL])

bad_labeled = labeled_train[~labeled_train[CAT_ID_COL].isin(category_tree.leaf_nodes)]  # Non-leaf category samples
good_labeled = labeled_train[labeled_train[CAT_ID_COL].isin(category_tree.leaf_nodes)]  # Leaf category samples

# Save bad_labeled data to parquet and free memory
bad_labeled.to_parquet(BAD_LABELED_PATH, index=False)
del bad_labeled

# Identify categories with only 1 sample and exclude them from good_labeled
cat_id_samples_cnt = good_labeled[CAT_ID_COL].value_counts()
one_sample_cats = cat_id_samples_cnt[cat_id_samples_cnt == 1].index.values
good_labeled = good_labeled[~good_labeled[CAT_ID_COL].isin(one_sample_cats)]

# Split data into initial train and out-of-sample (oos) sets
train_idx, test_idx = next(
    GroupKFold(n_splits=int(1 / TEST_PART_SIZE), shuffle=True, random_state=RANDOM_STATE).split(
        good_labeled, groups=good_labeled[CAT_ID_COL]
    )
)

# Further split train into train and in-sample (is) validation sets (stratified)
df_train, df_oos = good_labeled.iloc[train_idx], good_labeled.iloc[test_idx]
df_train, df_is = train_test_split(
    df_train, test_size=TEST_PART_SIZE, stratify=df_train[CAT_ID_COL], random_state=RANDOM_STATE
)

# Label part types for in-sample (is) and out-of-sample (oos) validation sets
df_is[PART_TYPE_COL] = "is"
df_oos[PART_TYPE_COL] = "oos"

# Concatenate is and oos sets into a single validation set
df_val = pd.concat([df_is, df_oos], axis=0)
val_hash_ids = df_val[HASH_ID_COL].values.tolist()

# Prepare and save validation dataset to parquet
df_val[PART_COL] = "val"
df_val = df_val[[HASH_ID_COL, TITLE_COL, CAT_ID_COL, PART_COL, PART_TYPE_COL]]
df_val.to_parquet(VAL_DATASET_PATH, index=False)

# Free memory
del labeled_train, df_train, one_sample_cats, df_val, df_is, df_oos, train_idx, test_idx, cat_id_samples_cnt

# Step 2: Split unlabeled_special_prize into good_prize and bad_prize
# Filter special prize data into samples with valid (leaf) category labels

unlabeled_special_prize = pd.read_parquet(UNLABELED_SPECIAL_PRIZE_PATH, columns=[TITLE_COL, CAT_ID_COL, HASH_ID_COL, ATTR_COL])

# bad_prize creation is commented out, assuming not needed further
# bad_prize = unlabeled_special_prize[~unlabeled_special_prize[CAT_ID_COL].isin(category_tree.leaf_nodes)]  # Non-leaf category samples (unused)
good_prize = unlabeled_special_prize[unlabeled_special_prize[CAT_ID_COL].isin(category_tree.leaf_nodes)]  # Leaf category samples

# Free memory
del unlabeled_special_prize

# Step 3: Remove good_prize samples from unlabeled_train
# Avoid duplicating samples between good_prize and unlabeled_train

unlabeled_train = pd.read_parquet(UNLABELED_PATH, columns=[TITLE_COL, HASH_ID_COL, ATTR_COL])

# Filter out samples from unlabeled_train that are already in good_prize (by hash ID)
unlabeled_train = unlabeled_train[~unlabeled_train[HASH_ID_COL].isin(good_prize[HASH_ID_COL])]

# Step 4: Merge good_labeled and good_prize datasets
# Combine all samples with valid (leaf) category labels

# Concatenate good_labeled (from Step 1) with good_prize (special prize data with leaf categories)
good_labeled = pd.concat([good_labeled, good_prize])

del good_prize

# Step 5: Create unlabeled_join by matching unlabeled_train with good_labeled via attributes
# Pseudo-label unlabeled samples by leveraging attribute-based category mappings from good_labeled

good_labeled_temp = good_labeled.copy()
good_labeled_temp[TITLE_COL] = good_labeled_temp[TITLE_COL].str.lower()

# Determine the most common category ID for each unique title in good_labeled
most_common_cat_id = good_labeled_temp.groupby(TITLE_COL)[CAT_ID_COL].agg(
    lambda x: x.value_counts().index[0] if len(x) > 0 else None
).reset_index()
most_common_cat_id.rename(columns={CAT_ID_COL: 'most_common_cat_id'}, inplace=True)

# Map most common category IDs back to good_labeled_temp
good_labeled_temp = good_labeled_temp.merge(most_common_cat_id, on=TITLE_COL, how='left')
good_labeled_temp[CAT_ID_COL] = good_labeled_temp['most_common_cat_id']

good_labeled_temp.drop(columns=['most_common_cat_id'], inplace=True)

unlabeled_join = unlabeled_train.copy()

# Free memory
del unlabeled_train

# Standardize attribute text (lowercase) in both datasets for matching
good_labeled_temp[ATTR_COL] = good_labeled_temp[ATTR_COL].str.lower()
unlabeled_join[ATTR_COL] = unlabeled_join[ATTR_COL].str.lower()

# Remove duplicate titles in both datasets (keep one representative row)
good_labeled_temp = good_labeled_temp.drop_duplicates([TITLE_COL])
unlabeled_join = unlabeled_join.drop_duplicates([TITLE_COL])

# Filter out rows with empty attributes ("[{}]") in both datasets
good_labeled_temp = good_labeled_temp[good_labeled_temp[ATTR_COL] != "[{}]"]
unlabeled_join = unlabeled_join[unlabeled_join[ATTR_COL] != "[{}]"]

# Join unlabeled_join with good_labeled_temp on attributes to pseudo-label categories
unlabeled_join = unlabeled_join.merge(
    good_labeled_temp[[ATTR_COL, CAT_ID_COL]], 
    on=ATTR_COL, 
    how='left'
).dropna(subset=[CAT_ID_COL])

# Free memory
del good_labeled_temp

# Resolve category ID conflicts for titles with multiple attribute-matched categories
unlabeled_join[TITLE_COL] = unlabeled_join[TITLE_COL].str.lower()
most_common_cat_id = unlabeled_join.groupby(TITLE_COL)[CAT_ID_COL].agg(lambda x: x.mode().iloc[0]).reset_index()
most_common_cat_id.rename(columns={CAT_ID_COL: 'most_common_cat_id'}, inplace=True)

# Assign the final, most common category ID per title
unlabeled_join = unlabeled_join.merge(most_common_cat_id, on=TITLE_COL, how='left')
unlabeled_join[CAT_ID_COL] = unlabeled_join['most_common_cat_id']
unlabeled_join.drop(columns=['most_common_cat_id'], inplace=True)
unlabeled_join = unlabeled_join.drop_duplicates([TITLE_COL])

# Final cleanup: deduplicate titles and drop attribute column (no longer needed
unlabeled_join.drop(columns=[ATTR_COL], inplace=True)
good_labeled.drop(columns=[ATTR_COL], inplace=True)

# Free memory
del most_common_cat_id

# Step 6: Load additional datasets to enrich training data
# Load unlabeled_dorazmetka - a subset of unlabeled data annotated by LLM (category 460)
unlabeled_dorazmetka = pd.read_parquet(UNLABELED_DORAZMETKA_PATH, columns=[TITLE_COL, HASH_ID_COL, CAT_ID_COL])

# Load new_items dataset, fully generated by LLM from scratch, and rename 'item_name' column to match TITLE_COL
new_items = pd.read_parquet(NEW_ITEMS_PATH).rename(columns={'item_name': TITLE_COL})

# Step 7: Concatenate datasets to form the final training set
# Exclude validation set items from good_labeled dataset
good_labeled = good_labeled[~good_labeled[HASH_ID_COL].isin(val_hash_ids)]

# Combine datasets into a single training set
train = pd.concat([good_labeled, unlabeled_join, unlabeled_dorazmetka, new_items], ignore_index=True, copy=False)
train[PART_TYPE_COL] = "is"
train[PART_COL] = "train"

# Remove duplicate items based on HASH_ID_COL
train.drop_duplicates(subset=[HASH_ID_COL], inplace=True)
# Select only relevant columns for training
train = train[[TITLE_COL, CAT_ID_COL, PART_COL, PART_TYPE_COL]]

# Shuffle the training set for better model generalization
train = train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Save the final training dataset to parquet file
train.to_parquet(TRAIN_DATASET_PATH)