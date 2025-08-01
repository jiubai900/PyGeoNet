import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from utils.download.tool import log_config, standard_gse, download_and_unzip, download_suppl, unzip_unrar, standard_gpl, download_gpl


def soft(gse, save_path, log_path, thread=4):
    """
    Download and unzip the soft files of GSE datasets.

    Parameters:
    - gse: List of GSE dataset identifiers.
    - save_path: Directory path to store downloaded files, default is the current directory.
    - thread: Number of threads to download concurrently, default is 4.

    Returns:
    None
    """
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(log_path):
        log_path = os.path.abspath(log_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = log_config(log_path, 'soft')
    # Standardize the list of GSE identifiers
    gse = standard_gse(gse)

    # Use a ThreadPoolExecutor to download and unzip files concurrently
    with ThreadPoolExecutor(max_workers=thread) as executor:
        # Iterate over each GSE identifier, construct the download URL, and submit the download task
        for index, content in enumerate(gse):
            # Determine the number part of the GSE identifier based on its length
            if len(content) > 6:
                number = content[0:-3]
            else:
                number = content[0:3]
            # Construct the download URL
            url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{number}nnn/{content}/soft'
            # Concatenate the save path
            path = os.path.join(save_path, content, 'soft')
            # Submit the download and unzip task to the thread pool
            executor.submit(download_and_unzip, url, content, path, logger)
    # Wait for all threads to complete their tasks
    executor.shutdown(wait=True)
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def matrix(gse, save_path, log_path, thread=4):
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(log_path):
        log_path = os.path.abspath(log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = log_config(log_path, 'matrix')
    gse = standard_gse(gse)
    with ThreadPoolExecutor(max_workers=thread) as executor:
        for index, content in enumerate(gse):
            if len(content) > 6:
                number = content[0:-3]
            else:
                number = content[0:3]
            url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{number}nnn/{content}/matrix'
            path = os.path.join(save_path, content, 'matrix')
            executor.submit(download_and_unzip, url, content, path, logger)
    executor.shutdown(wait=True)
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def suppl(gse, save_path, log_path, thread=4):
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(log_path):
        log_path = os.path.abspath(log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = log_config(log_path, 'suppl')
    gse = standard_gse(gse)
    with ThreadPoolExecutor(max_workers=thread) as executor:
        for index, content in enumerate(gse):
            if len(content) > 6:
                number = content[0:-3]
            else:
                number = content[0:3]
            url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{number}nnn/{content}/suppl/'
            path = os.path.join(save_path, content, 'suppl')
            executor.submit(download_suppl, url, content, path, logger)
    executor.shutdown(wait=True)
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
    unzip_unrar(save_path)


def all(gse, save_path, log_path, thread=4):
    soft(gse, save_path, log_path, thread)
    matrix(gse, save_path, log_path, thread)
    suppl(gse, save_path, log_path, thread)


def gpl(gpl_arr, save_path, log_path, thread=4):
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(log_path):
        log_path = os.path.abspath(log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = log_config(log_path, 'gpl')
    gpl_arr = standard_gpl(gpl_arr, logger)
    with ThreadPoolExecutor(max_workers=thread) as executor:
        for index, content in enumerate(gpl_arr):
            path = os.path.join(save_path, content)
            if not os.path.exists(path):
                os.makedirs(path)
            executor.submit(download_gpl, content, path, logger)
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def down_again(log_file, save_path, thread=4):
    """
    Re-download GSE files that failed to download previously based on the log file content.

    :param log_file: Path to the error log file
    :param save_path: Target directory to save downloaded files
    :param thread: Number of threads to use for downloading, default is 8
    """
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(log_file):
        log_file = os.path.abspath(log_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialize a set to store GSE IDs that encountered errors
    gse_arr = set()
    # Define a mapping to select the appropriate processing function based on file type
    file_operation_map = {
        'soft': soft,
        'matrix': matrix,
        'suppl': suppl,
        'gpl': gpl
    }
    try:
        # Read the error log file, analyze it, and extract GSE IDs that encountered errors
        with open(log_file, 'r') as file:
            for line in file:
                if 'ERROR' in line:
                    parts = line.split('!')
                    gse_arr.add(str(parts[1].strip()))
        # Clear the error log file
        with open(log_file, 'w') as file:
            file.write('')  # Explicitly write empty content

    except IOError as e:
        # If an error occurs during file operations, print the error message and exit the function
        print(f"File {log_file} operation failed: {e}")
        return
    # Convert the set to a list for subsequent processing
    gse_arr = list(set(gse_arr))
    # Get the filename to determine which processing function to call
    file_name = os.path.basename(log_file)
    # Select the appropriate processing function based on keywords in the filename
    for keyword in file_operation_map:
        if keyword in file_name:
            # Assume these functions handle multi-threading parameters
            file_operation_map[keyword](gse_arr, save_path, os.path.dirname(log_file), thread)
            break  # Stop searching after finding a match


def parse_args():
    parser = argparse.ArgumentParser(description="Process GSE datasets.")

    # This is where you set the parameters you need, including command and gse_arr.
    parser.add_argument('method', choices=['soft', 'matrix', 'suppl', 'all', 'gpl', 'down_again'],
                        help="Specify the command to execute.")

    parser.add_argument('--gse_arr', nargs='+',
                        help="Comma-separated GSE series numbers (e.g., 'GSE3,GSE11151,GSE1234')")

    parser.add_argument('--save_path', type=str, required=True, help="Path to save files.")
    parser.add_argument('--log_path', type=str, required=True, help="Path to save logs.")
    parser.add_argument('--thread', type=int, default=4, help="Number of threads for downloading.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Convert the --gse_arr argument from a string to a list

    # Choose which function to call based on the --command parameter passed.
    if args.method == 'soft':
        gse_arr = ','.join(args.gse_arr).split(',')
        soft(gse_arr, args.save_path, args.log_path, args.thread)
    elif args.method == 'matrix':
        gse_arr = ','.join(args.gse_arr).split(',')
        matrix(gse_arr, args.save_path, args.log_path, args.thread)
    elif args.method == 'suppl':
        gse_arr = ','.join(args.gse_arr).split(',')
        suppl(gse_arr, args.save_path, args.log_path, args.thread)
    elif args.method == 'all':
        gse_arr = ','.join(args.gse_arr).split(',')
        all(gse_arr, args.save_path, args.log_path, args.thread)
    elif args.method == 'gpl':
        gse_arr = ','.join(args.gse_arr).split(',')
        gpl(gse_arr, args.save_path, args.log_path, args.thread)
    elif args.method == 'down_again':
        down_again(args.log_path, args.save_path, args.thread)
    else:
        print("Invalid command")


if __name__ == "__main__":
    main()
