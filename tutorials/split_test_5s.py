from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets
from datasets import DatasetDict, Dataset

@dataclass
class SplittingDataset:
    dataset: Dataset = None
    strip_file_comparison: callable = None
    strip_site_comparison: callable = None


def get_splitting_dataset(name: str) -> SplittingDataset:
    """
    Downloads or loads cached Birdset Dataset with the given name.
    """
    dataset = load_dataset(
    name=name,
    path="DBD-research-group/BirdSet",
    cache_dir="../../data_birdset/" + name
    )

    strip_functions = map_stripping_functions(name)

    return SplittingDataset(dataset["test_5s"], strip_file_comparison=strip_functions[0], strip_site_comparison=strip_functions[1])


def map_stripping_functions(dataset_name: str) -> tuple[callable,callable]:
    """
    Maps the dataset name to filepath stripping functions that fit the dataset.
    Always returns a file stripping functions but only return a site stripping function for
    datasets that include site information in the filepath.
    """
    if (dataset_name == "NES_scape" 
        or dataset_name == "HSN_scape"
        or dataset_name == "POW_scape"
        or dataset_name == "SSW_scape"
        or dataset_name == "NBP_scape"
        or dataset_name == "SNE_scape"):
        return default_file_stripping, None
    elif (dataset_name == "PER_scape" 
          or dataset_name == "UHH_scape"):
        return default_file_stripping, default_site_stripping
    else:
        print(f"No stripping functions found for {dataset_name}. You'll have to set them manually")
        return None, None


def cut_underscores(path: str, num: int) -> str:
    """
    cuts till 'num' underscores from end of 'path'
    """
    for i in range(num):
        path = path[:path.rfind("_")]
    return path


def default_file_stripping(path :str) -> str:
    """
    The default filepath stripping to get the file identifier. 
    """
    path = cut_underscores(path, 2)
    path = path[path.rfind("\\")+1:]
    return path


def default_site_stripping(path: str) -> str:
    """
    The default filepath stripping to get the site identifier. 
    """
    site = cut_underscores(path, 4)
    site = site[-3:]
    return site


def split_into_sites(splitting_dataset: SplittingDataset) -> dict[str, Dataset]:
    """
    Splits the given dataset into datasets belonging to sites found in the original dataset.
    The sites are found and distinguished based on the given strip_site_comparison function which should
    strip the site descriptor out of the filepath.
    """
    if not splitting_dataset.strip_site_comparison:
        print("Site splitting function is not defined in given dataset")
        return


    dataset = splitting_dataset.dataset
    strip_site_comparison = splitting_dataset.strip_site_comparison
    sites = {}
    last_site = strip_site_comparison(dataset["filepath"][0])
    last_split_idx = 0

    all_files = dataset["filepath"]
    for idx in range(len(all_files)):
        site = strip_site_comparison(all_files[idx])

        if site != last_site:
            old_set = sites.get(last_site, None)
            new_set = dataset.select(range(last_split_idx, idx))
            if old_set:
                new_set = concatenate_datasets([old_set, new_set])
            sites[last_site] = new_set
            last_split_idx = idx
            last_site = site

    old_set = sites.get(last_site, None)
    new_set = dataset.select(range(last_split_idx, idx))
    if old_set:
        new_set = concatenate_datasets([old_set, new_set])
    sites[last_site] = new_set
    
    return sites


def split_dataset(splitting_dataset: SplittingDataset, split_from_idx : int, desired_test_split: float) -> DatasetDict:
    """
    Splits the given dataset into train and test splits. 
    The test split starts at the index given by "split_from_idx" and ends when the desired test split is achieved.
    Because of the fact that files should not be split apart the test is not always exactly as big as desired but
    it the nearest possibile percentage that doesn't split files.
    Files are distinguished by the given strip_file_comparison function which should strip the wanted file descriptor out of the filepath.
    """

    if not splitting_dataset.strip_file_comparison:
        print("File splitting function is not defined in given dataset")
        return

    dataset = splitting_dataset.dataset
    strip_file_comparison = splitting_dataset.strip_file_comparison

    num_rows = len(dataset)

    # find start of test split
    bottom_start_idx = split_from_idx
    top_start_idx = num_rows-1
    split_file = strip_file_comparison(dataset[split_from_idx]["filepath"])

    for idx in range(split_from_idx-1, -1, -1):
        file_at_idx = strip_file_comparison(dataset[idx]["filepath"])

        if file_at_idx != split_file:
            bottom_start_idx = idx + 1
            break

    for idx in range(split_from_idx+1, num_rows):
        filepath_at_idx = dataset[idx]["filepath"]
        file_at_idx = strip_file_comparison(filepath_at_idx)

        if file_at_idx != split_file:
            top_start_idx = idx
            break

    if split_from_idx - bottom_start_idx > top_start_idx - split_from_idx:
        nearest_start_idx = top_start_idx
    else:
        nearest_start_idx = bottom_start_idx

    # find end of test split
    desired_end_idx = min(nearest_start_idx + int(num_rows * desired_test_split), num_rows - 1)
    split_file = strip_file_comparison(dataset[desired_end_idx]["filepath"])

    bottom_end_idx = desired_end_idx
    top_end_idx = num_rows

    for idx in range(desired_end_idx-1, -1, -1):
        file_at_idx = strip_file_comparison(dataset[idx]["filepath"])

        if idx <= nearest_start_idx:
            bottom_end_idx = nearest_start_idx
            break

        if file_at_idx != split_file:
            bottom_end_idx = idx + 1
            break

    for idx in range(desired_end_idx, num_rows):
        file_at_idx = strip_file_comparison(dataset[idx]["filepath"])

        if file_at_idx != split_file:
            top_end_idx = idx
            break


    bottom_test_split = ((bottom_end_idx - nearest_start_idx) / num_rows)
    top_test_split = ((top_end_idx - nearest_start_idx) / num_rows)

    if desired_test_split - bottom_test_split > top_test_split - desired_test_split:
        nearest_end_idx = top_end_idx
    else:
        nearest_end_idx = bottom_end_idx

    # build datasets and dataset dict
    first_train_split = dataset.select(range(nearest_start_idx))
    test_split = dataset.select(range(nearest_start_idx, nearest_end_idx))
    if nearest_end_idx != num_rows:
        second_train_split = dataset.select(range(nearest_end_idx, num_rows))
    else:
        second_train_split = dataset.select(range(0))

    train_split = concatenate_datasets([first_train_split, second_train_split])
    dataset_dict = DatasetDict({'train':train_split, 'test':test_split})
    return dataset_dict


def split_into_k_datasets(splitting_dataset: SplittingDataset, k: int) -> list[DatasetDict]:
    """
    Splits the given dataset into k datasets of about equal test percentage (1/k)
    """
    if not splitting_dataset.strip_file_comparison:
        print("File splitting function is not defined in given dataset")
        return

    test_percentage_per_set = 1/k
    dataset_length = len(splitting_dataset.dataset)
    dataset_dicts = []

    for i in range(k):
        split_from_idx = int(dataset_length * (test_percentage_per_set * i))
        dataset_dict = split_dataset(splitting_dataset, split_from_idx, test_percentage_per_set)
        dataset_dicts.append(dataset_dict)

    return dataset_dicts


def split_into_k_datasets_with_sites(splitting_dataset: SplittingDataset, k: int) -> list[DatasetDict]:
    """
    Splits the given dataset into k datasets of about equal test percentage (1/k) while also trying to maintain site diversity in sets.
    """
    if not splitting_dataset.strip_file_comparison:
        print("File splitting function is not defined in given dataset")
        return
    if not splitting_dataset.strip_site_comparison:
        print("Site splitting function is not defined in given dataset")
        return
    
    dataset_sites: dict = split_into_sites(splitting_dataset)
    
    site_sets = {}
    for site, site_set in dataset_sites.items():
        site_splitting_set = SplittingDataset(
            dataset=site_set,
            strip_file_comparison=splitting_dataset.strip_file_comparison
            )
        site_set_dicts = split_into_k_datasets(site_splitting_set, k)
        site_sets[site] = site_set_dicts

    dataset_dicts = []
    for i in range(k):
        train_split = concatenate_datasets([site_sets[site][i]["train"] for site in site_sets])
        test_split = concatenate_datasets([site_sets[site][i]["test"] for site in site_sets])
        dataset_dict = DatasetDict({'train':train_split, 'test':test_split}) 
        dataset_dicts.append(dataset_dict)


    return dataset_dicts