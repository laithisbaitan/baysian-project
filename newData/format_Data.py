import pandas as pd
from newspaper import Article
from _change_text_to_summery import TextToSummery


def getData(url, title):
    try:
        art = Article(url)
        art.download()
        art.parse()
        text = art.text
        Summary = TextToSummery(text)
        source = art.source_url
        authors = art.authors
        # Reorder the columns for DataFrame
        data = [[source, authors, title, Summary]]
        df = pd.DataFrame(
            data, columns=['source', 'authors', 'title', 'summary'])
        return df
    except:
        return None


# Read the CSV file
input_file = 'newData\\archive_23\\ClaimFakeCOVID-19_5.csv'
df_csv = pd.read_csv(input_file)

# Create an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['source', 'authors', 'title', 'summary'])

# Loop through each row in the CSV and process the URLs
for index, row in df_csv.iterrows():
    url = row['news_url']
    title = row['title']
    print("Processing URL:", url)
    processed_data = getData(url, title)
    if processed_data is not None:
        result_df = pd.concat([result_df, processed_data], ignore_index=True)

# Save the result DataFrame to a new CSV file
result_df.to_csv('CovidOutput.csv', index=False)
