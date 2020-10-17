import mmcv
import os.path as osp

data_root = 'data/ReCTS/'
train_img_root = osp.join(data_root, 'train/img/')
train_ann_root = osp.join(data_root, 'train/gt/')
test_img_root = osp.join(data_root, 'test/img/')


def prepare_train_img_infos(cache_path, img_list_path=None):
    if img_list_path is not None:
        img_names = mmcv.list_from_file(img_list_path)
        img_names = [img_name + '.jpg' for img_name in img_names]
    else:
        img_names = [img_name for img_name in mmcv.utils.scandir(train_img_root, '.jpg')]

    img_infos = []
    print('Loading images...')
    for i, img_name in enumerate(img_names):
        if i % 1000 == 0:
            print('%d / %d' % (i, len(img_names)))
        img_path = train_img_root + img_name
        ann_file = img_name.replace('.jpg', '.json')

        try:
            h, w, _ = mmcv.imread(img_path).shape
            img_info = dict(
                filename=img_name,
                height=h,
                width=w,
                annfile=ann_file)
            img_infos.append(img_info)
        except:
            print('Load image error when generating img_infos: %s' % img_path)

    with open(cache_path, 'w') as f:
        mmcv.dump(img_infos, f, file_format='json', ensure_ascii=False)


def prepare_test_img_infos(cache_path):
    img_names = [img_name for img_name in mmcv.utils.scandir(test_img_root, '.jpg')]

    img_infos = []
    print('Loading images...')
    for i, img_name in enumerate(img_names):
        if i % 1000 == 0:
            print('%d / %d' % (i, len(img_names)))
        img_path = test_img_root + img_name
        ann_file = None

        try:
            h, w, _ = mmcv.imread(img_path).shape
            img_info = dict(
                filename=img_name,
                height=h,
                width=w,
                annfile=ann_file)
            img_infos.append(img_info)
        except:
            print('Load image error when generating img_infos: %s' % img_path)

    with open(cache_path, 'w') as f:
        mmcv.dump(img_infos, f, file_format='json', ensure_ascii=False)


def prepare_char_dict(img_list_path):
    img_names = mmcv.list_from_file(img_list_path)
    img_names = [img_name + '.jpg' for img_name in img_names]

    dictmap = mmcv.load(data_root + 'dictmap_to_lower.json')
    char2label = dict()

    chars = set()
    print('loading chars...')
    for i, img_name in enumerate(img_names):
        if i % 1000 == 0:
            print('%d / %d' % (i, len(img_names)))
        ann_path = train_ann_root + img_name.replace('.jpg', '.json')
        try:
            if ann_path is not None:
                ann_info = mmcv.load(ann_path)
                for i, ann in enumerate(ann_info['chars']):
                    char = ann['transcription']
                    if len(char) != 1:
                        continue
                    if char[0] == '#':
                        continue
                    if char in dictmap:
                        char = dictmap[char]
                    chars.add(char)
        except:
            print('Load ann error when generating char dict: %s' % ann_path)

    chars = list(chars)
    list.sort(chars)
    print('char num: %d' % len(chars))
    label2char = chars
    for i, char in enumerate(chars):
        char2label[char] = i + 1

    char_dict = {'char2label': char2label, 'label2char': label2char}
    with open(osp.join(data_root, 'char_dict.json'), 'w') as f:
        mmcv.dump(char_dict, f, file_format='json', ensure_ascii=False)


# def tmp(file_path):
#     img_infos = mmcv.load(file_path)
#     for i in range(len(img_infos)):
#         annpath = img_infos[i]['annpath']
#         annpath = annpath.split('/')[-1]
#         img_infos[i]['annfile'] = annpath
#         del img_infos[i]['annpath']
#
#     with open(file_path + '1', 'w') as f:
#         mmcv.dump(img_infos, f, file_format='json', ensure_ascii=False)


if __name__ == '__main__':
    # prepare img infos
    prepare_train_img_infos(osp.join(data_root, 'tda_rects_train_cache_file.json'),
                            osp.join(data_root, 'TDA_ReCTS_train_list.txt'))
    prepare_train_img_infos(osp.join(data_root, 'tda_rects_val_cache_file.json'),
                            osp.join(data_root, 'TDA_ReCTS_val_list.txt'))
    prepare_test_img_infos(osp.join(data_root, 'tda_rects_test_cache_file.json'))

    # # combine img infos
    img_infos = mmcv.load(osp.join(data_root, 'tda_rects_train_cache_file.json')) + \
                mmcv.load(osp.join(data_root, 'tda_rects_val_cache_file.json'))
    with open(osp.join(data_root, 'train_cache_file.json'), 'w') as f:
        mmcv.dump(img_infos, f, file_format='json', ensure_ascii=False)

    # # prepare char dict
    prepare_char_dict(osp.join(data_root, 'TDA_ReCTS_train_list.txt'))
