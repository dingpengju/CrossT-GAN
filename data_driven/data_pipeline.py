from torch.utils.data import DataLoader
from data_driven.data_builder import Dataset_MSL, Dataset_SMAP, Dataset_SMD, Dataset_SWaT

map_dict = {
    'MSL': Dataset_MSL,
    'SMAP': Dataset_SMAP,
    'SMD': Dataset_SMD,
    'SWaT': Dataset_SWaT,
}


def data_driven(args, flag):
    if flag == 'test':
        shuffle_flag = False
        batch_size = 1
    else:
        shuffle_flag = True
        batch_size = args.batch_size

    data_reader = map_dict[args.data_reader]
    data_set = data_reader(
        args=args,
        flag=flag
    )

    data_builder = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=True
    )
    args.logger.info('Mode: {}, Sample Num: {}, Batch Num: {}'.format(flag, len(data_set), len(data_builder)))
    return data_set, data_builder
