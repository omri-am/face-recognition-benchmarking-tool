import re

def get_function_name(func):
    if hasattr(func, '__name__') and func.__name__ != '<lambda>':
        return func.__name__
    else:
        return str(func)

def sort_mixed_column(df, column_name, additional_sort_fields = [], sort_key_order=None):
    def split_text_number(value):
        match = re.match(r"([a-zA-Z]+)(\d+)", value)
        if match:
            text_part = match.group(1)
            number_part = int(match.group(2))
            return text_part, number_part
        else:
            return value, 0
    sort_key_order = sort_key_order if sort_key_order else len(additional_sort_fields)
    df['sort_key'] = df[column_name].apply(lambda x: split_text_number(x))
    sort_by = additional_sort_fields[:sort_key_order] + ['sort_key'] + additional_sort_fields[sort_key_order:]
    df_sorted = df.sort_values(by=sort_by)
    df_sorted = df_sorted.drop(columns=['sort_key'])
    
    return df_sorted