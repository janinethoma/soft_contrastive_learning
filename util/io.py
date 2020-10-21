import csv
import pickle
import tarfile

import cv2
import numpy as np


# images
def save_img(img, out_file):
    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_file), img)


def load_img(in_file):
    img = cv2.imread(str(in_file))
    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# .txt
def load_txt(in_file):
    with open(in_file, 'r') as f:
        return f.read()


def save_txt(txt, out_file, mode='w'):
    with open(out_file, mode) as f:
        f.write(txt)


# .pickle
def load_pickle(in_file):
    with open(in_file, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


# .cvs
def load_csv(in_file, delimiter=',', has_header=True, keys=[]):
    """
    Reads data from csv into dict of lists
    :param in_file: path to file
    :param delimiter: default ','
    :param has_header: Does first row contain column names? Must be set to True for files with one line only.
    :param keys:
    :return: dict of lists with data from file
    """
    with open(in_file) as f:
        out_data = {}
        data = csv.reader(f, delimiter=delimiter)
        first = True
        has_multiple_rows = False
        for row in data:
            if first:
                first = False

                # Get keys (from header, given or initialize as np.arange(num_col))
                if has_header:
                    keys = [e for e in row]
                elif not len(keys) == len(row):
                    keys = np.arange(len(row))

                # Initialize empty lists
                for key in keys:
                    out_data[key] = []

                if has_header:
                    continue

            has_multiple_rows = True
            for i, key in enumerate(keys):
                out_data[key].append(row[i])
        if has_multiple_rows:
            return out_data
        else:
            return keys


def save_csv(data, out_file, delimiter=','):
    """
    :param data: Dict of header to list or single value (lists must have equal length)
    :param out_file: Path to output file
    :param delimiter:
    :return:
    """
    columns = list(data.keys())
    lines = list()
    lines.append(delimiter.join(['{}'.format(col) for col in columns]))

    if isinstance(data[columns[0]], list):
        for row in range(len(data[columns[0]])):
            row_data = ['{}'.format(data[col][row]) for col in columns]
            lines.append(delimiter.join(row_data))
    else:
        body = ['{}'.format(data[col]) for col in columns]
        lines.append(delimiter.join(body))

    save_txt('\n'.join(lines), out_file)


def unzip(in_file, out_file):
    tar = tarfile.open(in_file, mode="r")
    tar.extractall(out_file)
    tar.close()
