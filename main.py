import numpy as np
import pandas as pd
import rapidfuzz
from rapidfuzz import process
import re
import fuzzy_pandas as fpd

if __name__ == '__main__':
    threshold = 87

    # very important to assign proper delimiters, quotechar and escape char
    fb_df = pd.read_csv('resources/facebook_dataset.csv', delimiter=',', quotechar='"', escapechar="\\")
    gg_df = pd.read_csv('resources/google_dataset.csv', delimiter=',', quotechar='"', escapechar="\\")
    wb_df = pd.read_csv('resources/website_dataset.csv', delimiter=';')

    # lowercase all names and strip them of llc,LLC,llc., and same variations for inc, ltd
    # convert phone number to string
    fb_df['name'] = fb_df['name'].str.lower()
    fb_df['name'] = fb_df['name'].map(
        lambda x: re.sub(r'(?i)(llc\.\s*$|llc\s*$|inc\.\s*$|inc\s*$|ltd\.\s*$|ltd\s*$)', '', str(x)))
    fb_df['phone'] = fb_df['phone'].astype(str).str.replace('\\.0', '', regex=True)

    gg_df['name'] = gg_df['name'].str.lower()
    gg_df['name'] = gg_df['name'].map(
        lambda x: re.sub(r'(?i)(llc\.\s*$|llc\s*$|inc\.\s*$|inc\s*$|ltd\.\s*$|ltd\s*$)', '', str(x)))
    gg_df['phone'] = gg_df['phone'].astype(str).str.replace('\\.0', '', regex=True)

    wb_df['legal_name'] = wb_df['legal_name'].str.lower()
    wb_df['legal_name'] = wb_df['legal_name'].map(
        lambda x: re.sub(r'(?i)(llc\.\s*$|llc\s*$|inc\.\s*$|inc\s*$|ltd\.\s*$|ltd\s*$)', '', str(x)))
    wb_df['phone'] = wb_df['phone'].astype(str).str.replace('\\.0', '', regex=True)

    # deduplicate datasets based on name by keeping the row with the least NaN values
    gg_deduplicated_df = gg_df.loc[gg_df.notnull().sum(1).groupby(gg_df.name).idxmax()]
    fb_deduplicated_df = fb_df.loc[fb_df.notnull().sum(1).groupby(fb_df.name).idxmax()]
    wb_deduplicated_df = wb_df.loc[wb_df.notnull().sum(1).groupby(wb_df.legal_name).idxmax()]


    # this was used to investigate proper matching strategy
    def find_match(x):
        match = process.extract(x, fb_deduplicated_df['name'], limit=1, scorer=rapidfuzz.fuzz.ratio)[0]
        match = match if match[1] > threshold else np.nan
        return match


    # example usage of above function
    # gg_deduplicated_df['match found', 'score', 'index_position'] = gg_deduplicated_df['name'].apply(find_match)

    # rename the columns so we can compare them to confirm we have indeed matched the data properly
    gg_deduplicated_df.columns = ['gg_address', 'gg_category', 'gg_city', 'gg_country_code', 'gg_country_name',
                                  'gg_name',
                                  'gg_phone', 'gg_phone_country_code', 'gg_raw_address', 'gg_raw_phone',
                                  'gg_region_code', 'gg_region_name', 'gg_text', 'gg_zip_code', 'gg_domain']

    fb_deduplicated_df.columns = ['fb_domain', 'fb_address', 'fb_categories', 'fb_city', 'fb_country_code',
                                  'fb_country_name', 'fb_description', 'fb_email', 'fb_link', 'fb_name', 'fb_page_type',
                                  'fb_phone', 'fb_phone_country_code', 'fb_region_code', 'fb_region_name',
                                  'fb_zip_code']

    gg_fb_df = fpd.fuzzy_merge(gg_deduplicated_df, fb_deduplicated_df,
                               left_on='gg_name',
                               right_on='fb_name',
                               method='levenshtein',
                               ignore_nonalpha=True,
                               threshold=0.85)

    # This is how I was checking how many NaN values we have
    # print(gg_fb_df["gg_domain"].isna().sum())
    # print(gg_fb_df["fb_domain"].isna().sum())

    # This is how I was exporting my matches to check how they look
    # gg_fb_df.to_csv('google-matches-fb.csv', index=False)

    results1 = fpd.fuzzy_merge(gg_fb_df, wb_deduplicated_df,
                               left_on='gg_domain',
                               right_on='root_domain',
                               ignore_nonalpha=True,
                               threshold=0.95
                               )

    print(results1.head())

    # After this we should create a new dataframe with only the columns we intend to keep.
    # I have described which columns and which strategy for merging I would use in a doc accompanying my application

    print("Finished.")
