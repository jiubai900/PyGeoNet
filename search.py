import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
import ast
from bs4 import BeautifulSoup
from openpyxl.workbook import Workbook
from utils.search.tool import handle_all_fields, get_webpage_content

EXCEL_FILENAME = 'GEO.xlsx'
COLUMN_NAMES = {
    'A': 'number',
    'B': 'Status',
    'C': 'title',
    'D': 'Organism',
    'E': 'Experiment type',
    'F': 'Summary',
    'G': 'Overall design',
    'H': 'Submission date',
    'I': 'Last update date',
    'J': 'Citation',
    'K': 'Country',
    'L': 'Platforms',
    'M': '#Samples'
}


def contents(gse_arr, save_path='./', output_format='xlsx', thread=8):
    """
     Processes the contents of the GSE array and saves it to the specified path according to the specified output format.

     参数:
     - gse_arr: A GSE array, either a single string or a list of strings, representing the data source identifier to be processed.
     - save_path: Save path, default is current directory '. /'.
     - output_format: Output format, default is 'xlsx'. Optional values include 'csv', 'excel', 'df', 'txt'.
     - thread: Number of concurrent threads, default is 8.

     return value:
     - If output_format is 'df' or 'DataFrame', the processed DataFrame object is returned.
     """
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    # Check that the output format is valid
    format_arr = ['xlsx', 'excel', 'df', 'txt']
    if output_format not in format_arr:
        print("Invalid output format. Please choose , 'xlsx', 'excel', 'txt', or 'df'.")

    # Make sure gse_arr is of type list

    if gse_arr[0][-3::] == 'txt':
        with open(gse_arr[0], 'r', encoding='utf-16') as file:
            gse_arr = file.read().splitlines()
        if isinstance(gse_arr[0], str):
            gse_arr = gse_arr[0]
            gse_arr = ast.literal_eval(gse_arr)

    if type(gse_arr) == str:
        gse_arr = [str(gse_arr)]
    # Creating a Save Path
    os.makedirs(save_path, exist_ok=True)
    excel_filename = os.path.join(save_path, EXCEL_FILENAME)

    # Initialising an Excel workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "汇总"
    # Adding Column Names
    for col, name in COLUMN_NAMES.items():
        ws[col + '1'] = name

    # Using a thread pool to perform content fetching tasks
    with ThreadPoolExecutor(max_workers=thread) as executor:
        for index, content in enumerate(gse_arr):
            executor.submit(get_webpage_content, content, index, ws)

    # Wait for all threads to complete
    executor.shutdown()

    # Saving a workbook
    wb.save(excel_filename)

    # Further processing and saving based on output format
    if (output_format == 'xlsx') or (output_format == 'excel'):
        # Read data from Excel, clean and resave it
        df = pd.read_excel(excel_filename, index_col=0)
        df.dropna(how='all', inplace=True)
        df.to_excel(excel_filename)

        print(f"xlsx file saved at {excel_filename}")
    elif output_format == 'df' or output_format == 'DataFrame':
        # Read data from Excel, clean and return DataFrame
        df = pd.read_excel(excel_filename, index_col=0)
        df.dropna(how='all', inplace=True)
        df.to_excel(excel_filename)
        return df
    elif output_format == 'txt':
        # Read data from Excel, convert to txt format, clean up and save while deleting Excel file
        df = pd.read_excel(excel_filename, index_col=0)
        df.dropna(how='all', inplace=True)
        df.to_csv(f'{excel_filename[0:-4]}txt', sep='\t')
        os.remove(excel_filename)
        print(f"txt file saved at {excel_filename[0:-4]}txt")


def gse(ALL=None, AUTH=None, GTYP=None, DESC=None, ETYP=None, FILT=None,
        ACCN=None, MESH=None, NPRO=None, NSAM=None, ORGN=None, PTYP=None,
        PRO=None, PDAT=None, RGPL=None, RGSE=None, GEID=None, SRC=None,
        STYP=None, VTYP=None, INST=None, SSDE=None, SSTP=None, SFIL=None,
        TAGL=None, TITL=None, UDAT=None, retmax=5000):
    """
        Constructs and sends NCBI Gene Expression Omnibus (GEO) database query requests to retrieve records that meet specified criteria.

        parameters:
        - ALL: String that specifies all the fields used in the query.
        - AUTH: String specifying the author of the study.
        - GTYP: String specifying the sample type.
        - DESC: String specifying a brief description of the study.
        - ETYP: String specifying the type of experiment.
        - FILT: String specifying the conditions used to filter the results.
        - ACCN: String specifying the access number.
        - MESH: String specifying the MeSH term.
        - NPRO: Integer specifying the number of items.
        - NSAM: Integer specifying the number of samples.
        - ORGN: String specifying the organisation.
        - PTYP: String specifying the platform type.
        - PRO: String specifying the item ID.
        - PDAT: String specifying the date of submission.
        - RGPL: String specifying the library for the platform.
        - RGSE: String specifying the associated GSE number.
        - GEID: String specifying the GEO entry ID.
        - SRC: String specifying the source of the data.
        - STYP: String specifying the type of series.
        - VTYP: String specifying the data type.
        - INST: String specifying the organisation.
        - SSDE: String specifying the secondary study description.
        - SSTP: String specifying the sample status.
        - SFIL: String specifying the series filter.
        - TAGL: String specifying the length of the label.
        - TITL: String specifying the title of the study.
        - UDAT: String specifying the update date.
        - retmax: Integer, specifies the maximum number of records to be returned, defaults to 5000.
        For specific naming rules, please refer to  https://www.ncbi.nlm.nih.gov/geo/info/qqtutorial.html
        Return Value.
        - gse_arr: List containing formatted GEO series numbers.

        """
    field_key_value = {
        'ALL': ALL,
        'AUTH': AUTH,
        'GTYP': GTYP,
        'DESC': DESC,
        'ETYP': ETYP,
        'FILT': FILT,
        'ACCN': ACCN,
        'MESH': MESH,
        'NPRO': NPRO,
        'NSAM': NSAM,
        'ORGN': ORGN,
        'PTYP': PTYP,
        'PRO': PRO,
        'PDAT': PDAT,
        'RGPL': RGPL,
        'RGSE': RGSE,
        'GEID': GEID,
        'SRC': SRC,
        'STYP': STYP,
        'VTYP': VTYP,
        'INST': INST,
        'SSDE': SSDE,
        'SSTP': SSTP,
        'SFIL': SFIL,
        'TAGL': TAGL,
        'TITL': TITL,
        'UDAT': UDAT
    }
    # Define all possible field aliases
    field_aliases = ['ALL', 'AUTH', 'GTYP', 'DESC', 'ETYP', 'FILT', 'ACCN', 'MESH', 'NPRO', 'NSAM', 'ORGN',
                     'PTYP', 'PRO', 'PDAT', 'RGPL', 'RGSE', 'GEID', 'SRC', 'STYP', 'VTYP', 'INST', 'SSDE',
                     'SSTP', 'SFIL', 'TAGL', 'TITL', 'UDAT']
    term = 'term='
    i = 0
    # Constructing query strings
    for item in field_aliases:
        if item == 'ALL':
            term = handle_all_fields(term, item, field_key_value)
            i += 1
            continue

        if field_key_value[item]:
            if i == 0:
                term += f'{field_key_value[item]} [{item}]'
                i += 1
            else:
                term += '+AND+' + f'{field_key_value[item]} [{item}]'

    # Build and send HTTP requests
    http_url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&{term}&retmax={retmax}&usehistory=y'
    response = requests.get(http_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'xml')
        try:
            # Parsing the response to extract the GEO series number
            text = soup.find_all('Id')
            gse_number = [element.get_text(strip=True) for element in text]
            gse_arr = []
            for item in gse_number:
                if len(item) > 5:
                    number = item[3::]
                    while number[0] == '0':
                        number = number[1::]
                gse_arr.append(f'GSE{number}')
            return gse_arr
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return


def main():
    parser = argparse.ArgumentParser(description='Command-line interface to process and query GSE data.')

    subparsers = parser.add_subparsers(dest='command')

    # Subcommand for contents function
    contents_parser = subparsers.add_parser('contents', help='Process GSE content and save in specified format.')
    contents_parser.add_argument('--gse_arr', type=str, nargs='+', required=True,
                                 help='Space-separated GSE series numbers (e.g., "GSE11151 GSE3 GSE1234").')
    contents_parser.add_argument('--save_path', type=str, default='./',
                                 help='Directory to save output. Default is current directory.')
    contents_parser.add_argument('--output_format', type=str, choices=['xlsx', 'excel', 'df', 'txt'], default='xlsx',
                                 help='Output format: "xlsx", "excel", "df", or "txt".')
    contents_parser.add_argument('--thread', type=int, default=8,
                                 help='Number of threads for processing. Default is 8.')

    # Arguments for the gse function
    gse_parser = subparsers.add_parser('gse', help='Query GEO database with specified parameters.')
    gse_parser.add_argument('--ALL', type=str, nargs='+', help='Multiple search terms for the "ALL" field (e.g.,"cancer", "tumor").')
    gse_parser.add_argument('--AUTH', type=str, help='Author name(s) for filtering the search.')
    gse_parser.add_argument('--GTYP', type=str, help='Sample type (e.g., "tumor", "normal").')
    gse_parser.add_argument('--DESC', type=str, help='Description term for the search.')
    gse_parser.add_argument('--ETYP', type=str, help='Experiment type (e.g., "microarray", "RNA-seq").')
    gse_parser.add_argument('--FILT', type=str, help='Additional filters for the search (e.g., "humans").')
    gse_parser.add_argument('--ACCN', type=str, help='Accession number to search for.')
    gse_parser.add_argument('--MESH', type=str, help='Medical Subject Headings (MeSH) term for filtering.')
    gse_parser.add_argument('--NPRO', type=str, help='Number of probes (e.g., "1000").')
    gse_parser.add_argument('--NSAM', type=str, help='Number of samples in the dataset.')
    gse_parser.add_argument('--ORGN', type=str, help='Organism name (e.g., "Homo sapiens").')
    gse_parser.add_argument('--PTYP', type=str, help='Platform type (e.g., "Illumina").')
    gse_parser.add_argument('--PRO', type=str, help='Specific probe or gene identifier.')
    gse_parser.add_argument('--PDAT', type=str, help='Publication date for filtering.')
    gse_parser.add_argument('--RGPL', type=str, help='Platform used in the experiment.')
    gse_parser.add_argument('--RGSE', type=str, help='GSE series number for a specific experiment.')
    gse_parser.add_argument('--GEID', type=str, help='Gene expression identifier.')
    gse_parser.add_argument('--SRC', type=str, help='Source of the dataset.')
    gse_parser.add_argument('--STYP', type=str, help='Sample type (e.g., "control", "treated").')
    gse_parser.add_argument('--VTYP', type=str, help='Data type (e.g., "raw", "processed").')
    gse_parser.add_argument('--INST', type=str, help='Institution conducting the experiment.')
    gse_parser.add_argument('--SSDE', type=str, help='Sample source designation (e.g., "public").')
    gse_parser.add_argument('--SSTP', type=str, help='Sample type (e.g., "RNA").')
    gse_parser.add_argument('--SFIL', type=str, help='File format (e.g., "txt", "csv").')
    gse_parser.add_argument('--TAGL', type=str, help='Tag or keyword for the dataset.')
    gse_parser.add_argument('--TITL', type=str, help='Title of the experiment.')
    gse_parser.add_argument('--UDAT', type=str, help='Update date for the experiment data.')

    # Argument for the number of results to return
    gse_parser.add_argument('--retmax', type=int, default=5000,
                        help='Maximum number of results to return from the search (default is 5000).')

    # Parse the arguments from the command line
    args = parser.parse_args()

    if args.command == 'contents':
        contents(gse_arr=args.gse_arr, save_path=args.save_path, output_format=args.output_format, thread=args.thread)
    elif args.command == 'gse':
        gse_results = gse(ALL=args.ALL, AUTH=args.AUTH, GTYP=args.GTYP, DESC=args.DESC, ETYP=args.ETYP,
                          FILT=args.FILT, retmax=args.retmax)
        print(gse_results)


if __name__ == '__main__':
    main()
