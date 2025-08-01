import logging
from typing import Union, List
from datetime import datetime
import gzip
import os
import shutil
import tarfile
import time
import pandas as pd
from GEOparse import get_GEO
from bs4 import BeautifulSoup
import requests
import tqdm

# Assume get_GEO function is already defined and does not need modification

# Define constants
NULL_VALUE = 'null'
GPL_ID_SEPARATOR = ' '


def standard_gse(gse: Union[str, List[str]]) -> List[str]:
    """
    Standardize gse input to ensure a list of strings is returned.
    """
    # If input is a list, return it directly
    if isinstance(gse, list):
        if gse and (gse[0][-3::] == 'csv' or gse[0][-4::] == 'xlsx'):
            gse = gse[0]
        else:
            return gse

    # If input is a string
    if isinstance(gse, str):
        # If it is a file path
        if os.path.isfile(gse):
            # Determine file type and read it
            file_ext = os.path.splitext(gse)[1]
            file_type = file_ext[1:].lower()
            if file_type in ['csv', 'xlsx']:
                gse = read_file_to_list(gse, file_type)
            else:
                print(f"Unsupported file extension: {file_ext}")
                gse = []
        # If it is a single string, put it in a list and return
        else:
            gse = [gse]

    # Ensure the final return is a list type
    return gse


def read_file_to_list(file_path: str, file_type: str) -> List[str]:
    """
    Read a file and return the first column as a list based on the file type.
    """
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'xlsx':
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return df[df.columns[0]].tolist()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def log_config(log_path, log_name):
    """
    Configure the logger.

    This function initializes and configures the logger to ensure that logs are recorded according to the specified path and name.
    If the specified log file does not exist, a new log file will be created.

    Parameters:
    log_path (str): The storage path for the log file.
    log_name (str): The prefix name for the log file.

    Returns:
    logging.Logger: A configured logger object.
    """
    # Get the Logger object for the current module
    logger = logging.getLogger(__name__)
    # Set the logging level to INFO
    logger.setLevel(logging.INFO)

    # Get the current date and time to generate the log file name
    # Check if FileHandler already exists to avoid duplication
    now = datetime.now()
    # Format the date as year-month-day
    formatted_date = now.strftime('%Y-%m-%d')
    # Construct the full path for the log file
    log_path = log_path + '\\' + formatted_date + '-' + log_name + '-logger' + '.log'
    # Check if a FileHandler for the specified path already exists
    if not any(
            isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path for handler in
            logger.handlers):
        # If not, create a new FileHandler
        file_handler = logging.FileHandler(log_path)
        # Configure the log format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # Add the FileHandler to the Logger object
        logger.addHandler(file_handler)

    # Return the configured logger
    return logger


def safe_download_file(file_url, save_path, local_filename, logger, gse_number):
    """
    Safely download a file.

    Parameters:
    file_url: str - The URL of the file.
    save_path: str - The local path to save the file.
    local_filename: str - The name to save the file as.

    Returns:
    None.
    """
    # Validate input parameters
    if not file_url.startswith('http') or not os.path.isabs(save_path):
        logger.error("Invalid input parameters.")
        return

    retry_count = 3  # Define retry count
    timeout = 10  # Set timeout to 10 seconds

    # Start download attempts
    for i in range(retry_count):
        try:
            response = requests.get(file_url, timeout=timeout)
            # Download successful
            if response.status_code == 200:
                # Create save path
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # Save file
                with open(os.path.join(save_path, local_filename), 'wb') as f:
                    f.write(response.content)
                logger.info(f'Successfully downloaded {local_filename}')
                break
            # Download failed, log error and decide whether to retry
            else:
                logger.error(
                    f'!{gse_number}!Failed to download {local_filename}, server returned status code: {response.status_code}')
                if i == retry_count - 1:
                    logger.error(f'!{gse_number}!Exceeded maximum retry count for file {local_filename}')
                else:
                    time.sleep(3)  # Wait 3 seconds before the next retry
        # Handle connection errors
        except requests.exceptions.ConnectionError:
            logger.error(f'!{gse_number}!Connection error while downloading {local_filename}')
            if i == retry_count - 1:
                logger.error(f'!{gse_number}!Exceeded maximum retry count for file {local_filename}')
            else:
                time.sleep(3)  # Wait 3 seconds before the next retry
        except requests.exceptions.Timeout:
            logger.error(f'!{gse_number}!Timeout error while downloading {local_filename}')
            if i == retry_count - 1:
                logger.error(f'!{gse_number}!Exceeded maximum retry count for file {local_filename}')
            else:
                time.sleep(3)  # Wait 3 seconds before the next retry
        except Exception as e:
            logger.error(f'!{gse_number}!An unexpected error occurred while downloading {local_filename}: {e}')
            break


def download_and_unzip(url, gse_number, save_path, logger):
    """
    Download and unzip files from the specified URL.

    Parameters:
    - url: The URL to download and unzip files from.
    - gse_number: The specific GSE number for logging.
    - save_path: The target path to save and unzip files.

    Returns:
    None.
    """
    try:
        logger.info(f'Starting download  GSE number {gse_number}')
        code_443 = False  # Flag to mark if 443 error is encountered
        response = requests.get(url)  # Try to get the URL content
        if response.status_code == 200:  # Request successful
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.find_all('a')  # Find all links

            # Iterate over all found links, try to download and unzip gz files
            for p in tqdm.tqdm(text, desc='Downloading files'):
                _, ext = os.path.splitext(p.text)  # Get file extension
                if ext.lower() in ['.gz']:  # If it's a gz file
                    r_path = f'{url}/{p.text}'  # Build the full file download path
                    safe_download_file(r_path, save_path, p.text, logger, gse_number)  # Safely download the file
                    if os.path.exists(f'{save_path}/{p.text}'):  # If download is successful, try to unzip
                        with gzip.open(f'{save_path}/{p.text}', 'rb') as f_gz, open(f'{save_path}/{p.text[0:-3]}',
                                                                                    'wb') as f_out:
                            try:
                                shutil.copyfileobj(f_gz, f_out)  # Copy file stream to unzip
                                logger.info(f'Successfully unzipped and saved {p.text[0:-3]}')
                            except Exception as e:
                                logger.error(f'!{gse_number}!Error while processing {p.text}: {e}')
                                continue  # Process the next file
                        os.remove(f'{save_path}/{p.text}')  # Delete the unzipped gz file
                    else:
                        code_443 = True

            if code_443:  # If 443 error is encountered, move the entire directory
                shutil.move(f'{save_path}', f'{save_path}_443')

        else:  # Handle request failure
            logger.error(f'!{gse_number}!Request for {url} failed with status code {response.status_code}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)  # Ensure the directory exists
            shutil.move(f'{save_path}', f'{save_path}_443')  # Move the directory
    except Exception as e:
        logger.error(f'!{gse_number}!An unexpected error occurred: {e}')  # Log unexpected errors


def handle_file_extension(filepath, save_path, filename):
    """
    Process files based on their file extensions.

    Parameters:
    - filepath: The path of the original file.
    - save_path: The target path to save the extracted or decompressed files.
    - filename: The filename, including the extension.

    Returns:
    None.
    """
    # Get the file extension and convert to lowercase
    ext1 = os.path.splitext(filename)[1].lower()
    if ext1 in ['.tar']:
        # If the file is a .tar, extract and delete the original file
        with tarfile.open(filepath, 'r') as tar:
            tar.extractall(save_path)
        os.remove(filepath)
    elif ext1 in ['.gz']:
        # If the file is a .gz, decompress and delete the original file
        with gzip.open(filepath, 'rb') as gz_file, open(os.path.join(save_path, filename[0:-3]), 'wb') as output_file:
            shutil.copyfileobj(gz_file, output_file)
        os.remove(filepath)
    for root, dirs, files in os.walk(save_path):
        for file in files:
            if file.endswith('.gz'):
                with gzip.open(os.path.join(root, file), 'rb') as gz_file, open(os.path.join(root, file[0:-3]),
                                                                                'wb') as output_file:
                    shutil.copyfileobj(gz_file, output_file)
                os.remove(os.path.join(root, file))
            if file.endswith('.tar'):
                with tarfile.open(os.path.join(root, file), 'r') as tar:
                    tar.extractall(root)
                os.remove(os.path.join(root, file))


def download_suppl(url, gse_number, save_path, logger):
    """
    Download supplementary data files for the specified GSE number.

    Parameters:
    url - The URL of the data source.
    gse_number - The GSE (Gene Expression Omnibus) number to identify the specific gene expression dataset.
    save_path - The local path to save the downloaded files.

    Returns:
    None.
    """
    try:
        logger.info(f'Starting download GSE number {gse_number}')  # Log the start of the download
        code_443 = False
        # Flag to indicate if a 443 error (i.e., file cannot be accessed or downloaded) is encountered）

        # Check if the save path exists, create it if it doesn't
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Attempt multiple downloads to improve stability
        for num_i in range(1, 4):
            try:
                response = requests.get(url)  # Try to get the web page content
                if response.status_code == 200:  # If the request is successful
                    soup = BeautifulSoup(response.text, 'html.parser')  # Parse HTML using BeautifulSoup
                    text = soup.find_all('a')  # Find all links
                    if text:
                        for item in text:
                            # Check if the link points to a .tar, .gz, or .xlsx file and try to download it
                            _, ext1 = os.path.splitext(f'{url}/{item.text}')
                            if ext1.lower() in ['.tar', '.gz', '.xlsx']:
                                r_path = f'{url}/{item.text}'
                                safe_download_file(r_path, save_path, item.text, logger, gse_number)
                                # After downloading, process the file based on its type
                                if os.path.exists(os.path.join(save_path, item.text)):
                                    handle_file_extension(os.path.join(save_path, item.text), save_path, item.text)
                                else:
                                    code_443 = True  # If the file cannot be saved properly, mark it as a 443 error
                        if code_443:
                            # If a 443 error is encountered, rename the save directory to _443 as a marker
                            shutil.move(f'{save_path}', f'{save_path}_443')
                        break
                elif response.status_code == 404:
                    break

                else:
                    logger.error(f'!{gse_number}!Request for {url} failed with status code {response.status_code}')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    shutil.move(f'{save_path}', f'{save_path}_443')

            except requests.exceptions.RequestException as e:
                # Handle exceptions that may occur during the request and try again after waiting for a while
                logger.error(f"!{gse_number}!Error occurred: {e}")
                time.sleep(num_i)
    except Exception as e:
        # Handle other unexpected exceptions
        logger.error(f'!{gse_number}!An unexpected error occurred: {e}')


def unzip_unrar(url):
    """
    Recursively unzip and unrar all files in the specified directory.

    Parameters:
    - url: The directory URL containing the files to be processed.

    Returns:
    None.
    """
    for root, dirs, files in os.walk(url):
        remove_arr = []  # List to store paths of files to be deleted

        # Process each file in the directory
        for file in files:
            file_path = os.path.join(root, file)  # Join the file path

            # Delete files smaller than 1KB
            if os.path.getsize(file_path) < 1024:
                remove_arr.append(file_path)

            # Process .tar files
            if file.endswith(('.tar')):
                with tarfile.open(file_path, 'r') as tar:
                    try:
                        tar.extractall(file_path[0:-4])  # Extract .tar files
                        remove_arr.append(file_path)  # Add to the list of files to be deleted
                        unzip_unrar(file_path[0:-4])  # Recursively process the extracted directory
                    except Exception as e:
                        print(f'error:{e}')  # Print the exception message

            elif file.endswith(('.gz')):
                with gzip.open(file_path, 'rb') as f_in:
                    with open(file_path[0:-3], 'wb') as f_out:
                        try:
                            shutil.copyfileobj(f_in, f_out)  # Decompress .gz files
                            remove_arr.append(file_path)  # Add to the list of files to be deleted
                            unzip_unrar(file_path[0:-3])  # Recursively process the decompressed file or directory
                        except Exception as e:
                            print(f'error:{e}')  # Print the exception message
        # Delete all files in the remove_arr list
        for item in remove_arr:
            os.remove(item)


def standard_gpl(gpl, logger):
    """
    Standardize the format of platform lists.

    Parameters:
    gpl - Can be a string, a list of strings, or a path to a file, where the file can be an Excel table.

    Returns:
    A standardized list of platform names. Each element in the list is a unique platform name, and duplicates have been removed.
    """
    # If the input is a list, return the list directly
    if isinstance(gpl, list):
        if gpl[0][-4::] == 'xlsx':
            gpl = gpl[0]
        else:
            return gpl

    # Handle the case where the input is a string
    if isinstance(gpl, str):
        # If the string is a file path
        if os.path.isfile(gpl):
            # Determine the file type based on the file extension and process accordingly
            file_ext = os.path.splitext(gpl)[1]
            file_type = file_ext[1:].lower()
            if file_type in ['xlsx']:
                # Read platform data from the Excel file
                df = pd.read_excel(gpl)
                gpl = df['Platforms'].tolist()
                gpl_set = set()
                # Split and process each row of data, add to the set to remove duplicates
                for item in gpl:
                    try:
                        gpl_item = item.split(' ')
                        for case in gpl_item:
                            gpl_set.add(case)
                    except:
                        continue
                gpl = list(gpl_set)  # Convert the set to a list

            else:
                # Log an error for unsupported file types
                logger.error(f"Unsupported file extension: {file_ext}")
                gpl = []
        # If the input string is not a file path, treat it as a single platform name and add it to the list
        else:
            gpl = [gpl]

        # Ensure the final return value is a list
    return gpl


def download_gpl(gpl, save_path, logger):
    """
    Attempt to download the specified GEO GPL dataset and save it to the specified path.

    Parameters:
    gpl (str): The identifier or ID of the GEO GPL dataset.
    save_path (str): The directory path to save the dataset file, default is the current directory.

    Returns:
    bool: Returns True if the file is successfully downloaded and saved, otherwise returns False.
    """
    # Initialize retry parameters
    max_retry = 3
    retry_wait_time = 1
    for attempt in range(max_retry):
        try:
            get_GEO(gpl, destdir=save_path)  # Try to download the dataset
            # Check if the file was successfully downloaded
            if os.path.exists(os.path.join(save_path, f'{gpl}.txt')):
                logger.info(f"Successfully downloaded {gpl}.txt")
                return True
            else:
                logger.warning(f"!{gpl}!Failed to download {gpl}.txt on attempt {attempt + 1}")
                time.sleep(retry_wait_time)  # Sleep and retry
                retry_wait_time *= 2  # Use exponential backoff to increase retry interval
        except Exception as e:
            logger.error(f"!{gpl}!An error occurred: {e}")
            time.sleep(retry_wait_time)
            retry_wait_time *= 2

    logger.error(f"!{gpl}!Max retry attempts ({max_retry}) exceeded for downloading {gpl}.txt")
    return False
