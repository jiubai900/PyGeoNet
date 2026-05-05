import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from utils.normalize.tool import sum_get_data, normalize_log


def get_data(raw_data_path, gpl_path='', save_path='./', thread=4, single_operation=False):
    """
    Get data from the specified path.

    Parameters:
    - raw_data_path (str): Path to the raw data, default is current directory './'.
    - gpl_path (str): GPL path, unused, reserved parameter.
    - save_path (str): Path to save the data, default is current directory './'.
    - thread (int): Number of threads to use, default is 4.
    - single_operation (bool): Whether to perform a single operation, default is False.

    Returns:
    No return value.
    """
    normalize_log("Normalize module started.")
    normalize_log(
        f"Input arguments: raw_data_path={raw_data_path}, gpl_path={gpl_path}, "
        f"save_path={save_path}, thread={thread}, single_operation={single_operation}"
    )

    if not raw_data_path:
        normalize_log("Path is empty. Stop normalize module.")
        return

    if not os.path.isabs(raw_data_path):
        raw_data_path = os.path.abspath(raw_data_path)
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(gpl_path):
        gpl_path = os.path.abspath(gpl_path)

    normalize_log(f"Resolved raw_data_path: {raw_data_path}")
    normalize_log(f"Resolved gpl_path: {gpl_path}")
    normalize_log(f"Resolved save_path: {save_path}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        normalize_log(f"Created save directory: {save_path}")

    if single_operation:
        normalize_log(f"Single-operation mode: processing {raw_data_path}")
        sum_get_data(raw_data_path, gpl_path, save_path)
        normalize_log("Normalize module finished in single-operation mode.")

    elif not single_operation:
        if os.path.exists(raw_data_path):
            dataset_dirs = []
            for root, dirs, files in os.walk(raw_data_path):
                dataset_dirs = [os.path.join(root, d) for d in dirs]
                break

            normalize_log(f"Batch mode: found {len(dataset_dirs)} first-level dataset directories.")

            if not dataset_dirs:
                normalize_log("No dataset directory found. Stop normalize module.")
                return

            max_workers = max(1, int(thread))
            normalize_log(f"Submitting normalize tasks with max_workers={max_workers}.")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(sum_get_data, dataset_dir, gpl_path, save_path): dataset_dir
                    for dataset_dir in dataset_dirs
                }

                for future in as_completed(futures):
                    dataset_dir = futures[future]
                    dataset_name = os.path.basename(dataset_dir)

                    try:
                        future.result()
                        normalize_log(f"Task finished: {dataset_name}")
                    except Exception as exc:
                        normalize_log(f"Task failed: {dataset_name}. Error: {exc}")

            normalize_log("Normalize module finished in batch mode.")
        else:
            normalize_log(f"Raw data path does not exist: {raw_data_path}")
            return

    else:
        normalize_log("Parameter error. Stop normalize module.")
        return


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Get and process raw data")

    # Add arguments to the parser
    parser.add_argument("raw_data_path",type=str, help="Path to the raw data")
    parser.add_argument("--gpl_path", help="Path to the GPL file", default="")
    parser.add_argument("--save_path", help="Path to save the processed data", default='./')
    parser.add_argument("--thread", help="Number of threads to use", type=int, default=4)
    parser.add_argument("--single_operation", help="Whether to perform a single operation",
                        action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Call the get_data function with the parsed arguments
    get_data(args.raw_data_path, args.gpl_path, args.save_path, args.thread, args.single_operation)


if __name__ == "__main__":
    main()
