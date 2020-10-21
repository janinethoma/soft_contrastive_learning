import argparse
import os
from collections import Counter
from collections import OrderedDict

from learnlarge.util.helper import fs_root
from learnlarge.util.io import load_csv, save_csv
from learnlarge.util.plot import dict_to_bar


def get_tags(tag_root):
    tags = dict()
    all_tags = []
    for date in os.listdir(tag_root):
        tags[date] = load_csv(os.path.join(tag_root, date, 'tags.csv'))
        all_tags = list(set(all_tags + tags[date]))
    return tags, all_tags


def plot_statistics(in_root, out_root, folds, tag_root):
    date_tags, all_tags = get_tags(tag_root)

    for fold in folds:
        print('Plotting {} statistics.'.format(fold))
        clean_file = os.path.join(in_root, '{}.csv'.format(fold))
        data = load_csv(clean_file)

        # Images per date
        images_per_date = Counter(data['date'])
        save_csv(images_per_date, os.path.join(out_root, 'images_per_date_{}.csv'.format(fold)))
        dict_to_bar(images_per_date, os.path.join(out_root, 'images_per_date_{}.pdf'.format(fold)))

        # Images/dates per tag, month and hour
        images_per_tag = dict.fromkeys(all_tags, 0)
        images_per_month = dict.fromkeys(range(1, 13), 0)
        images_per_hour = dict.fromkeys(range(0, 24), 0)

        dates_per_tag = dict.fromkeys(all_tags, 0)
        dates_per_month = dict.fromkeys(range(1, 13), 0)
        dates_per_hour = dict.fromkeys(range(0, 24), 0)

        for date in images_per_date.keys():
            month = int(date[5:7])
            hour = int(date[11:13])
            images_per_month[month] = images_per_date[date] + images_per_month[month]
            images_per_hour[hour] = images_per_date[date] + images_per_hour[hour]

            dates_per_month[month] = 1 + dates_per_month[month]
            dates_per_hour[hour] = 1 + dates_per_hour[hour]
            for tag in date_tags[date]:
                images_per_tag[tag] = images_per_date[date] + images_per_tag[tag]
                dates_per_tag[tag] = 1 + dates_per_tag[tag]

        save_csv(images_per_tag, os.path.join(out_root, 'images_per_tag_{}.csv'.format(fold)))
        dict_to_bar(images_per_tag, os.path.join(out_root, 'images_per_tag_{}.pdf'.format(fold)))

        save_csv(images_per_tag, os.path.join(out_root, 'images_per_tag_{}.csv'.format(fold)))
        dict_to_bar(images_per_tag, os.path.join(out_root, 'images_per_tag_{}.pdf'.format(fold)))

        save_csv(images_per_month, os.path.join(out_root, 'images_per_month_{}.csv'.format(fold)))
        dict_to_bar(images_per_month, os.path.join(out_root, 'images_per_month_{}.pdf'.format(fold)))

        new_months = OrderedDict()
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                  'November', 'December']
        for i in range(12):
            new_months[months[i]] = images_per_month[i + 1]

        save_csv(new_months, os.path.join(out_root, 'images_per_month_pretty_{}.csv'.format(fold)))
        dict_to_bar(new_months, os.path.join(out_root, 'images_per_month_pretty_{}.pdf'.format(fold)))

        save_csv(images_per_hour, os.path.join(out_root, 'images_per_hour_{}.csv'.format(fold)))
        dict_to_bar(images_per_hour, os.path.join(out_root, 'images_per_hour_{}.pdf'.format(fold)))

        new_hours = OrderedDict()
        for i in range(6, 22):
            new_hours['{:02d}:00'.format(i)] = images_per_hour[i]
        save_csv(new_hours, os.path.join(out_root, 'images_per_pretty_hour_{}.csv'.format(fold)))
        dict_to_bar(new_hours, os.path.join(out_root, 'images_per_pretty_hour_{}.pdf'.format(fold)))

        save_csv(dates_per_tag, os.path.join(out_root, 'dates_per_tag_{}.csv'.format(fold)))
        dict_to_bar(dates_per_tag, os.path.join(out_root, 'dates_per_tag_{}.pdf'.format(fold)))

        save_csv(dates_per_month, os.path.join(out_root, 'dates_per_month_{}.csv'.format(fold)))
        dict_to_bar(dates_per_month, os.path.join(out_root, 'dates_per_month_{}.pdf'.format(fold)))

        save_csv(dates_per_hour, os.path.join(out_root, 'dates_per_hour_{}.csv'.format(fold)))
        dict_to_bar(dates_per_hour, os.path.join(out_root, 'dates_per_hour_{}.pdf'.format(fold)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/clean_merged_parametrized'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/statistics'))
    parser.add_argument('--tag_root', default='/path/to/server/files/data/datasets/oxford_extracted')
    # parser.add_argument('--folds', default=['train', 'val', 'test', 'full', 'train_ref', 'val_ref', 'test_ref', 'full_ref', 'train_query', 'val_query', 'test_query', 'full_query'])
    parser.add_argument('--folds',
                        default=['train_ref'])
    args = parser.parse_args()

    in_root = args.in_root
    out_root = args.out_root
    tag_root = args.tag_root
    folds = args.folds

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    plot_statistics(in_root, out_root, folds, tag_root)


if __name__ == '__main__':
    main()
