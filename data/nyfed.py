import pandas as pd


def acmtp():
    """
    ACM Term Premium straight from the NY Fed's website
    """
    file_url = r"https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
    df = pd.read_excel(file_url, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

