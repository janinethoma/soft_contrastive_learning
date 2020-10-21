import os


def flags_to_globals(args):
    for elem in sorted(args.__dict__.keys()):
        print('{}=FLAGS.{}'.format(elem.upper(), elem))


def flags_to_args(args):
    for elem in sorted(args.__dict__.keys()):
        print('{}=args.{}'.format(elem, elem))


def fs_root():
    return '/path/to/my/files'


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
