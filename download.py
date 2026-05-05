import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

from utils.download.tool import (
    log_config,
    standard_gse,
    download_and_unzip,
    download_suppl,
    unzip_unrar,
    standard_gpl,
    download_gpl
)

from utils.log_tool import (
    create_task_logger,
    log_section,
    log_args,
    log_path_status,
    log_exception
)


def soft(gse, save_path, log_path, thread=4, task_id=None):
    """
    Download and unzip the soft files of GSE datasets.
    """
    logger, log_file = create_task_logger(
        module_name="download_soft",
        log_path=log_path,
        task_id=task_id
    )

    start_time = time.time()

    log_section(logger, "SOFT DOWNLOAD STARTED")
    log_args(
        logger,
        gse=gse,
        save_path=save_path,
        log_path=log_path,
        thread=thread,
        task_id=task_id
    )

    try:
        if not os.path.isabs(save_path):
            save_path = os.path.abspath(save_path)
        if not os.path.isabs(log_path):
            log_path = os.path.abspath(log_path)

        log_path_status(logger, "Resolved save_path", save_path)
        log_path_status(logger, "Resolved log_path", log_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logger.info(f"Created save directory: {save_path}")

        if not os.path.exists(log_path):
            os.makedirs(log_path)
            logger.info(f"Created log directory: {log_path}")

        gse = standard_gse(gse)

        if not gse:
            logger.warning("No valid GSE id found after standardization. SOFT download stopped.")
            log_section(logger, "SOFT DOWNLOAD FINISHED WITH EMPTY INPUT")
            logger.info(f"Log file saved at: {log_file}")
            return

        logger.info(f"Standardized GSE count: {len(gse)}")
        logger.info(f"Standardized GSE preview: {gse[:20]}")

        max_workers = max(1, int(thread))
        logger.info(f"Thread pool max_workers: {max_workers}")

        task_info = {}

        log_section(logger, "SUBMITTING SOFT DOWNLOAD TASKS")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, content in enumerate(gse, start=1):
                if len(content) > 6:
                    number = content[0:-3]
                else:
                    number = content[0:3]

                url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{number}nnn/{content}/soft'
                path = os.path.join(save_path, content, 'soft')

                logger.info(
                    f"[{index}/{len(gse)}] Submit SOFT task | "
                    f"GSE={content} | URL={url} | save_path={path}"
                )

                future = executor.submit(download_and_unzip, url, content, path, logger)
                task_info[future] = {
                    "gse": content,
                    "url": url,
                    "path": path
                }

            log_section(logger, "WAITING FOR SOFT DOWNLOAD TASKS")

            success_count = 0
            failed_count = 0

            for future in as_completed(task_info):
                info = task_info[future]
                content = info["gse"]
                path = info["path"]

                try:
                    future.result()
                    success_count += 1
                    logger.info(f"[{content}] SOFT task finished. Output path: {path}")
                except Exception as exc:
                    failed_count += 1
                    log_exception(logger, f"[{content}] SOFT task failed.", exc)

        elapsed = time.time() - start_time

        log_section(logger, "SOFT DOWNLOAD FINISHED")
        logger.info(f"Total submitted tasks: {len(gse)}")
        logger.info(f"Finished tasks: {success_count}")
        logger.info(f"Failed tasks: {failed_count}")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info(f"Log file saved at: {log_file}")

    except Exception as e:
        log_exception(logger, "Unexpected error occurred in soft download.", e)
        logger.info(f"Log file saved at: {log_file}")
        return


def matrix(gse, save_path, log_path, thread=4, task_id=None):
    """
    Download and unzip the matrix files of GSE datasets.
    """
    logger, log_file = create_task_logger(
        module_name="download_matrix",
        log_path=log_path,
        task_id=task_id
    )

    start_time = time.time()

    log_section(logger, "MATRIX DOWNLOAD STARTED")
    log_args(
        logger,
        gse=gse,
        save_path=save_path,
        log_path=log_path,
        thread=thread,
        task_id=task_id
    )

    try:
        if not os.path.isabs(save_path):
            save_path = os.path.abspath(save_path)
        if not os.path.isabs(log_path):
            log_path = os.path.abspath(log_path)

        log_path_status(logger, "Resolved save_path", save_path)
        log_path_status(logger, "Resolved log_path", log_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logger.info(f"Created save directory: {save_path}")

        if not os.path.exists(log_path):
            os.makedirs(log_path)
            logger.info(f"Created log directory: {log_path}")

        gse = standard_gse(gse)

        if not gse:
            logger.warning("No valid GSE id found after standardization. MATRIX download stopped.")
            log_section(logger, "MATRIX DOWNLOAD FINISHED WITH EMPTY INPUT")
            logger.info(f"Log file saved at: {log_file}")
            return

        logger.info(f"Standardized GSE count: {len(gse)}")
        logger.info(f"Standardized GSE preview: {gse[:20]}")

        max_workers = max(1, int(thread))
        logger.info(f"Thread pool max_workers: {max_workers}")

        task_info = {}

        log_section(logger, "SUBMITTING MATRIX DOWNLOAD TASKS")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, content in enumerate(gse, start=1):
                if len(content) > 6:
                    number = content[0:-3]
                else:
                    number = content[0:3]

                url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{number}nnn/{content}/matrix'
                path = os.path.join(save_path, content, 'matrix')

                logger.info(
                    f"[{index}/{len(gse)}] Submit MATRIX task | "
                    f"GSE={content} | URL={url} | save_path={path}"
                )

                future = executor.submit(download_and_unzip, url, content, path, logger)
                task_info[future] = {
                    "gse": content,
                    "url": url,
                    "path": path
                }

            log_section(logger, "WAITING FOR MATRIX DOWNLOAD TASKS")

            success_count = 0
            failed_count = 0

            for future in as_completed(task_info):
                info = task_info[future]
                content = info["gse"]
                path = info["path"]

                try:
                    future.result()
                    success_count += 1
                    logger.info(f"[{content}] MATRIX task finished. Output path: {path}")
                except Exception as exc:
                    failed_count += 1
                    log_exception(logger, f"[{content}] MATRIX task failed.", exc)

        elapsed = time.time() - start_time

        log_section(logger, "MATRIX DOWNLOAD FINISHED")
        logger.info(f"Total submitted tasks: {len(gse)}")
        logger.info(f"Finished tasks: {success_count}")
        logger.info(f"Failed tasks: {failed_count}")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info(f"Log file saved at: {log_file}")

    except Exception as e:
        log_exception(logger, "Unexpected error occurred in matrix download.", e)
        logger.info(f"Log file saved at: {log_file}")
        return


def suppl(gse, save_path, log_path, thread=4, task_id=None):
    """
    Download and unzip supplementary files of GSE datasets.
    """
    logger, log_file = create_task_logger(
        module_name="download_suppl",
        log_path=log_path,
        task_id=task_id
    )

    start_time = time.time()

    log_section(logger, "SUPPL DOWNLOAD STARTED")
    log_args(
        logger,
        gse=gse,
        save_path=save_path,
        log_path=log_path,
        thread=thread,
        task_id=task_id
    )

    try:
        if not os.path.isabs(save_path):
            save_path = os.path.abspath(save_path)
        if not os.path.isabs(log_path):
            log_path = os.path.abspath(log_path)

        log_path_status(logger, "Resolved save_path", save_path)
        log_path_status(logger, "Resolved log_path", log_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logger.info(f"Created save directory: {save_path}")

        if not os.path.exists(log_path):
            os.makedirs(log_path)
            logger.info(f"Created log directory: {log_path}")

        gse = standard_gse(gse)

        if not gse:
            logger.warning("No valid GSE id found after standardization. SUPPL download stopped.")
            log_section(logger, "SUPPL DOWNLOAD FINISHED WITH EMPTY INPUT")
            logger.info(f"Log file saved at: {log_file}")
            return

        logger.info(f"Standardized GSE count: {len(gse)}")
        logger.info(f"Standardized GSE preview: {gse[:20]}")

        max_workers = max(1, int(thread))
        logger.info(f"Thread pool max_workers: {max_workers}")

        task_info = {}

        log_section(logger, "SUBMITTING SUPPL DOWNLOAD TASKS")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, content in enumerate(gse, start=1):
                if len(content) > 6:
                    number = content[0:-3]
                else:
                    number = content[0:3]

                url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{number}nnn/{content}/suppl/'
                path = os.path.join(save_path, content, 'suppl')

                logger.info(
                    f"[{index}/{len(gse)}] Submit SUPPL task | "
                    f"GSE={content} | URL={url} | save_path={path}"
                )

                future = executor.submit(download_suppl, url, content, path, logger)
                task_info[future] = {
                    "gse": content,
                    "url": url,
                    "path": path
                }

            log_section(logger, "WAITING FOR SUPPL DOWNLOAD TASKS")

            success_count = 0
            failed_count = 0

            for future in as_completed(task_info):
                info = task_info[future]
                content = info["gse"]
                path = info["path"]

                try:
                    future.result()
                    success_count += 1
                    logger.info(f"[{content}] SUPPL task finished. Output path: {path}")
                except Exception as exc:
                    failed_count += 1
                    log_exception(logger, f"[{content}] SUPPL task failed.", exc)

        log_section(logger, "POST-PROCESSING SUPPL FILES")
        logger.info(f"Start recursive unzip/unrar for save_path: {save_path}")

        try:
            unzip_unrar(save_path)
            logger.info("Recursive unzip/unrar finished.")
        except Exception as exc:
            log_exception(logger, "Recursive unzip/unrar failed.", exc)

        elapsed = time.time() - start_time

        log_section(logger, "SUPPL DOWNLOAD FINISHED")
        logger.info(f"Total submitted tasks: {len(gse)}")
        logger.info(f"Finished tasks: {success_count}")
        logger.info(f"Failed tasks: {failed_count}")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info(f"Log file saved at: {log_file}")

    except Exception as e:
        log_exception(logger, "Unexpected error occurred in supplementary download.", e)
        logger.info(f"Log file saved at: {log_file}")
        return


def all(gse, save_path, log_path, thread=4, task_id=None):
    """
    Download soft, matrix, and supplementary files for GSE datasets.
    """
    logger, log_file = create_task_logger(
        module_name="download_all",
        log_path=log_path,
        task_id=task_id
    )

    start_time = time.time()

    log_section(logger, "ALL DOWNLOAD STARTED")
    log_args(
        logger,
        gse=gse,
        save_path=save_path,
        log_path=log_path,
        thread=thread,
        task_id=task_id
    )

    try:
        if task_id is None:
            task_id = os.path.basename(log_file).replace("download_all_", "").replace(".log", "")

        logger.info("Step 1/3: start SOFT download.")
        soft(gse, save_path, log_path, thread, task_id=f"{task_id}_soft")
        logger.info("Step 1/3: SOFT download finished.")

        logger.info("Step 2/3: start MATRIX download.")
        matrix(gse, save_path, log_path, thread, task_id=f"{task_id}_matrix")
        logger.info("Step 2/3: MATRIX download finished.")

        logger.info("Step 3/3: start SUPPL download.")
        suppl(gse, save_path, log_path, thread, task_id=f"{task_id}_suppl")
        logger.info("Step 3/3: SUPPL download finished.")

        elapsed = time.time() - start_time

        log_section(logger, "ALL DOWNLOAD FINISHED")
        logger.info("Download stages completed: soft, matrix, suppl")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info(f"Main log file saved at: {log_file}")

    except Exception as e:
        log_exception(logger, "Unexpected error occurred in all download.", e)
        logger.info(f"Main log file saved at: {log_file}")
        return


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
