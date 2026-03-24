import re

import xlsxwriter


def file_setup(filename: str) -> xlsxwriter.Workbook:
    return xlsxwriter.Workbook(filename)


def add_worksheet(workbook: xlsxwriter.Workbook, name: str) -> xlsxwriter.worksheet:
    return workbook.add_worksheet(name)


def file_closeout(workbook: xlsxwriter.Workbook) -> None:
    workbook.close()


def check_filename_ext(filename: str, ext: str) -> str:
    """
    If filename without extension is provided. Put in the extension
    """

    file_parts = re.split(r"\.", filename)
    ext_clean = re.split(r"\.", ext)[-1]
    if file_parts[-1] == ext_clean:
        return filename
    return f"{filename}.{ext_clean}"
