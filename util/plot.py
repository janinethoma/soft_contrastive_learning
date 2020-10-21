import matplotlib.pyplot as plt


def dict_to_bar(data, out_file):
    plt.clf()
    f = plt.figure()
    f.set_figheight(7)
    f.set_figwidth(7)
    names = list(data.keys())
    values = list(data.values())
    plt.bar(range(len(data)), values, tick_label=names)
    plt.xticks(rotation=45)
    plt.savefig(out_file)
    plt.close(f)
