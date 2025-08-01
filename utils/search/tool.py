from bs4 import BeautifulSoup
import httpx

# Define constants

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


def handle_all_fields(term, item, field_key_value):
    """Handle the special case for the 'ALL' field"""
    cases = field_key_value[item]
    for i, case in enumerate(cases):
        if i == 0:
            term += f'{case} [{item}]'
        else:
            term += '+OR+' + f'{case} [{item}]'
    return term


def get_webpage_content(content, index, ws):
    """
    Scrape and process data from a webpage, and fill it into an Excel sheet.

    Parameters:
    - content: The content to query (e.g., NCBI GEO database access ID)
    - index: The starting row index in the Excel sheet for the data
    - ws: The Excel worksheet object to write the scraped data into

    No return value; directly writes data into the provided worksheet object (ws).
    """

    # Print start information
    print(index, content, 'start')

    # Construct the URL
    url = f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={content}'
    http_content_dict = {}  # Dictionary to store the scraped content

    try:
        num_i = 1
        while num_i < 5:
            # Make an HTTP request
            response = httpx.get(url)
            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                trs = soup.find('table').find_all('tr', valign='top')

                if trs:
                    # Process HTML table data
                    for tr in trs:
                        rows = tr.find_all('td')
                        for i, row in enumerate(rows):
                            if i == 0:
                                http_content_dict[row.get_text()] = []
                            else:
                                http_content_dict[rows[0].get_text()].append(row.get_text())

                    # Extract and process Sample count information
                    sample_key = [k for k in http_content_dict.keys() if 'Sample' in k]
                    if sample_key:
                        num = str(sample_key).split('(')[1].split(')')[0].split(' ')
                        http_content_dict['Sample'] = num

                    # Extract Platform information
                    platform_key = [k for k in http_content_dict.keys() if 'Platforms' in k]
                    if platform_key:
                        pl = platform_key[0]
                        platforms = http_content_dict[pl][1:]
                        http_content_dict['Platform'] = [item for i, item in enumerate(platforms) if i % 2 == 0]

                    # Extract Organism information
                    organism_keys = [k for k in http_content_dict.keys() if 'Organism' in k] + \
                                    [k for k in http_content_dict.keys() if 'Sample organism' in k]
                    if organism_keys:
                        http_content_dict['Organisms'] = http_content_dict[organism_keys[0]]

                    # Extract Citation information
                    citation_key = [k for k in http_content_dict.keys() if 'Citation' in k]
                    if citation_key:
                        http_content_dict['Citation'] = http_content_dict[citation_key[0]]

                    # Default case, initialize 'Overall design' to an empty list if it does not exist
                    if 'Overall design' not in http_content_dict:
                        http_content_dict['Overall design'] = []

                    # Write the extracted data to the Excel sheet
                    keys = ['Status', 'Title', 'Organisms', 'Experiment type', 'Summary', 'Overall design',
                            'Submission date', 'Last update date', 'Citation', 'Country', 'Platform', 'Sample']
                    excel_numbers = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
                    for i, n in enumerate(excel_numbers):
                        try:
                            y_string = ' '.join(http_content_dict[keys[i]])
                            ws[f'{n}{index + 2}'] = y_string
                        except Exception as e:
                            print(f"{index} {content} error writing to Excel: {e}")

                    # Write the ID content at the beginning of the row
                    ws[f'A{index + 2}'] = f'{content}'
                    print(index, content, 'end')
                    break
                else:
                    # If no content is scraped, print a message and break the loop
                    print(index, content, 'no content')
                    break

            # Handle the case of request failure, especially 443 error (access denied)
            elif response.status_code == 443:
                ws[f'A{index + 2}'] = f'{content}(443)'
                print(f"{index} {content} request failed, status code: {response.status_code}")
                break
            else:
                # Retry logic for other types of errors
                num_i += 1
                print(f"{index} {content} request failed, status code: {response.status_code}")
    except httpx.RequestError as e:
        # Handle request errors
        print(f"Request error: {e}")
