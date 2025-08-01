import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from utils.normalize.tool import sum_get_data


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
    if not raw_data_path:
        print("Path is empty")
        return

    if not os.path.isabs(raw_data_path):
        raw_data_path = os.path.abspath(raw_data_path)
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(gpl_path):
        gpl_path = os.path.abspath(gpl_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if single_operation:
        sum_get_data(raw_data_path, gpl_path, save_path)

    elif not single_operation:
        # Check if the path exists
        if os.path.exists(raw_data_path):
            # Traverse all files and directories under the path
            # Use a thread pool to execute data retrieval tasks
            with ThreadPoolExecutor(max_workers=thread) as executor:
                for root, dirs, files in os.walk(raw_data_path):
                    # Submit a task for each directory
                    for d in dirs:
                        executor.submit(sum_get_data, os.path.join(root, d), gpl_path, save_path)
                    # Wait for all threads to complete
                    executor.shutdown(wait=True)
                    # Stop traversal after finding the first matching path
                    break

    else:
        print("Parameter error")
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
