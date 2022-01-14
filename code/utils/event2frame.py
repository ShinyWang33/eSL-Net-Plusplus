import numpy as np

def event_to_cnt_img(event, height=180, width=320):
    if len(event) == 0:
        return np.zeros((2, height, width), dtype=np.float32)
    t = event[:, 0]
    p = event[:, 1].astype(np.uint8)
    x = event[:, 2].astype(np.uint32)
    y = event[:, 3].astype(np.uint32)

    pos_event_x = x[p >= 1]
    pos_event_y = y[p >= 1]

    neg_event_x = x[p < 1]
    neg_event_y = y[p < 1]

    event_cnt_img_pos = np.zeros(height * width, dtype=np.float32)
    event_cnt_img_neg = np.zeros(height * width, dtype=np.float32)

    np.add.at(event_cnt_img_pos, pos_event_y * width + pos_event_x, 1)
    event_cnt_img_pos = event_cnt_img_pos.reshape([height, width])

    np.add.at(event_cnt_img_neg, neg_event_y * width + neg_event_x, 1)
    event_cnt_img_neg = event_cnt_img_neg.reshape([height, width])

    event_cnt_img = np.stack((event_cnt_img_pos, event_cnt_img_neg), 0)


    return event_cnt_img


def split_events_by_time(events, index_frame, sum_frame, split_num=20, start_ts=None, end_ts=None):
    """
    :param events: raw events [n, 4] t, p, x, y
    :param split_num: int
    :param index_frame: index of the reconstructed image for the total results of a blurry image
    :param sum_frame: the number of the reconstructed images for a blurry image
    """
    start_ts = events[0, 0] if start_ts is None else float(start_ts)
    end_ts = events[-1, 0] if end_ts is None else float(end_ts)

    split_ts_interval = (end_ts - start_ts)/split_num

    f_time = (end_ts - start_ts) * index_frame / (sum_frame-1) + start_ts

    f_idx = np.searchsorted(events[:, 0], f_time)


    event_shift = events[f_idx:, :]
    event_reversal = events[:f_idx, :]

    # event polarity inverse
    event_reversal_inverse = event_reversal.copy()
    event_reversal_p = event_reversal[:, 1]
    event_reversal_inverse[:, 1] = 1-event_reversal_p

    len_shift = len(event_shift[:, 0])
    len_reversal = len(event_reversal_inverse[:, 0])

    # event shift split index
    split_shift_idx_lst = [0]
    for i in range(1, split_num + 1):
        if len_shift == 0:
            split_shift_idx_lst.append(0)
            continue
        split_ts = f_time + i * split_ts_interval
        split_idx = np.searchsorted(event_shift[:, 0], split_ts, side='right')
        #if split_idx == len_shift:
        #    split_idx = split_idx - 1
        split_shift_idx_lst.append(split_idx)

    # event reversal split index
    split_reversal_idx_lst = [len_reversal]
    for i in range(1, split_num + 1):
        if len_reversal == 0:
            split_reversal_idx_lst.append(0)
            continue
        split_ts = f_time - i * split_ts_interval
        split_idx = np.searchsorted(event_reversal_inverse[:, 0], split_ts)
        #if split_idx == 0:
        #    split_idx = split_idx - 1
        split_reversal_idx_lst.append(split_idx)

    # event shift split
    events_lst = []
    for i in range(split_num):
        start_idx = split_shift_idx_lst[i]
        end_idx = split_shift_idx_lst[i + 1]
        events_split = event_shift[start_idx:end_idx, :]
        events_lst.append(events_split)

    # event reversal split
    for i in range(split_num):
        start_idx = split_reversal_idx_lst[i + 1]
        end_idx = split_reversal_idx_lst[i]
        events_split = event_reversal_inverse[start_idx:end_idx, :]
        events_lst.append(events_split)

    return events_lst
