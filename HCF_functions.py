import numpy as np

import cv2

def get_ball_position(obs=None):
    assert obs is not None
    assert len(obs.shape) == 3 and obs.shape[0] == 84 and obs.shape[1] == 84 and obs.shape[2] == 4

    h_offset = 11
    w_offset = 2
    obs = obs[h_offset:75, w_offset:82, :]

    curr = obs[:, :, 3]
    prev = obs[:, :, 2]

    curr_lap = np.abs(cv2.Laplacian(curr, cv2.CV_64F))
    prev_lap = np.abs(cv2.Laplacian(prev, cv2.CV_64F))

    diff_lap = abs(curr_lap - prev_lap)

    _, max_val, _, max_loc = cv2.minMaxLoc(diff_lap)

    loc = max_loc + np.array([w_offset, h_offset])
    loc_a = (int(np.floor(max_loc[0] / 8)), int(np.floor(max_loc[1] / 8)))

    areas = {
        '0': (0, 0),
        '1': (0, 1),
        '2': (0, 2),
        '3': (0, 3),
        '4': (0, 4),
        '5': (0, 5),
        '6': (0, 6),
        '7': (0, 7),
        '8': (1, 0),
        '9': (1, 1),
        '10': (1, 2),
        '11': (1, 3),
        '12': (1, 4),
        '13': (1, 5),
        '14': (1, 6),
        '15': (1, 7),
        '16': (2, 0),
        '17': (2, 1),
        '18': (2, 2),
        '19': (2, 3),
        '20': (2, 4),
        '21': (2, 5),
        '22': (2, 6),
        '23': (2, 7),
        '24': (3, 0),
        '25': (3, 1),
        '26': (3, 2),
        '27': (3, 3),
        '28': (3, 4),
        '29': (3, 5),
        '30': (3, 6),
        '31': (3, 7),
        '33': (4, 0),
        '34': (4, 1),
        '35': (4, 2),
        '36': (4, 3),
        '37': (4, 4),
        '38': (4, 5),
        '39': (4, 6),
        '40': (4, 7),
        '41': (5, 0),
        '42': (5, 1),
        '43': (5, 2),
        '44': (5, 3),
        '45': (5, 4),
        '46': (5, 5),
        '47': (5, 6),
        '48': (5, 7),
        '49': (6, 0),
        '50': (6, 1),
        '51': (6, 2),
        '52': (6, 3),
        '53': (6, 4),
        '54': (6, 5),
        '55': (6, 6),
        '56': (6, 7),
        '57': (7, 0),
        '58': (7, 1),
        '59': (7, 2),
        '60': (7, 3),
        '61': (7, 4),
        '62': (7, 5),
        '63': (7, 6),
        '64': (7, 7),
        '65': (8, 0),
        '66': (8, 1),
        '67': (8, 2),
        '68': (8, 3),
        '69': (8, 4),
        '70': (8, 5),
        '71': (8, 6),
        '72': (8, 7),
        '73': (9, 0),
        '74': (9, 1),
        '75': (9, 2),
        '76': (9, 3),
        '77': (9, 4),
        '78': (9, 5),
        '79': (9, 6),
        '80': (9, 7)
    }

    for k, v in areas.items():
        if loc_a[0] == v[0] and loc_a[1] == v[1]:
            area = k
            break
    return max_val, loc, area


def get_digit(obs=None, which=None, method=cv2.TM_CCOEFF_NORMED):
    """
    return the digit from raw image
    method: string in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                       cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
    """
    assert obs is not None
    assert obs.shape[0] == 4 and obs.shape[1] == 7
    assert obs.dtype in [np.uint8, np.float32]
    assert method in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                      cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
    assert which in ['hundreds', 'tens', 'ones', 'lives']

    if which == 'ones':  # units digit
        digits_filters = {
            '0': np.array([[43, 142, 136, 114, 116, 142, 142],
                           [43, 142, 114, 0, 14, 142, 142],
                           [43, 142, 114, 0, 14, 142, 142],
                           [43, 142, 136, 114, 116, 142, 142], ]),
            '1': np.array([[0, 0, 28, 142, 128, 0, 0],
                           [0, 0, 28, 142, 128, 0, 0],
                           [0, 0, 28, 142, 128, 0, 0],
                           [0, 0, 28, 142, 128, 0, 0], ]),
            '2': np.array([[34, 114, 114, 114, 116, 142, 142],
                           [17, 57, 57, 57, 65, 142, 142],
                           [43, 142, 125, 57, 57, 57, 57],
                           [43, 142, 136, 114, 114, 114, 114], ]),
            '3': np.array([[34, 114, 114, 114, 116, 142, 142],
                           [0, 0, 11, 57, 65, 142, 142],
                           [0, 0, 11, 57, 65, 142, 142],
                           [34, 114, 114, 114, 116, 142, 142], ]),
            '4': np.array([[43, 142, 114, 0, 14, 142, 142],
                           [43, 142, 125, 57, 65, 142, 142],
                           [17, 57, 57, 57, 65, 142, 142],
                           [0, 0, 0, 0, 14, 142, 142], ]),
            '5': np.array([[43, 142, 136, 114, 114, 114, 114],
                           [43, 142, 125, 57, 57, 57, 57],
                           [17, 57, 57, 57, 65, 142, 142],
                           [34, 114, 114, 114, 116, 142, 142], ]),
            '6': np.array([[43, 142, 114, 0, 0, 0, 0],
                           [43, 142, 125, 57, 57, 57, 57],
                           [43, 142, 125, 57, 65, 142, 142],
                           [43, 142, 136, 114, 116, 142, 142], ]),
            '7': np.array([[34, 114, 114, 114, 116, 142, 142],
                           [0, 0, 0, 0, 14, 142, 142],
                           [0, 0, 0, 0, 14, 142, 142],
                           [0, 0, 0, 0, 14, 142, 142], ]),
            '8': np.array([[43, 142, 136, 114, 116, 142, 142],
                           [43, 142, 125, 57, 65, 142, 142],
                           [43, 142, 125, 57, 65, 142, 142],
                           [43, 142, 136, 114, 116, 142, 142], ]),
            '9': np.array([[43, 142, 136, 114, 116, 142, 142],
                           [43, 142, 125, 57, 65, 142, 142],
                           [17, 57, 57, 57, 65, 142, 142],
                           [0, 0, 0, 0, 14, 142, 142], ])
        }
    elif which == 'tens':  # tens digit
        digits_filters = {
            '0': np.array([[99, 142, 125, 114, 128, 142, 85],
                           [99, 142, 57, 0, 71, 142, 85],
                           [99, 142, 57, 0, 71, 142, 85],
                           [99, 142, 125, 114, 128, 142, 85], ]),
            '1': np.array([[0, 0, 85, 142, 71, 0, 0],
                           [0, 0, 85, 142, 71, 0, 0],
                           [0, 0, 85, 142, 71, 0, 0],
                           [0, 0, 85, 142, 71, 0, 0], ]),
            '2': np.array([[80, 114, 114, 114, 128, 142, 85],
                           [40, 57, 57, 57, 99, 142, 85],
                           [99, 142, 91, 57, 57, 57, 34],
                           [99, 142, 125, 114, 114, 114, 68], ]),
            '3': np.array([[80, 114, 114, 114, 128, 142, 85],
                           [0, 0, 34, 57, 99, 142, 85],
                           [0, 0, 34, 57, 99, 142, 85],
                           [80, 114, 114, 114, 128, 142, 85], ]),
            '4': np.array([[99, 142, 57, 0, 71, 142, 85],
                           [99, 142, 91, 57, 99, 142, 85],
                           [40, 57, 57, 57, 99, 142, 85],
                           [0, 0, 0, 0, 71, 142, 85], ]),
            '5': np.array([[99, 142, 125, 114, 114, 114, 68],
                           [99, 142, 91, 57, 57, 57, 34],
                           [40, 57, 57, 57, 99, 142, 85],
                           [80, 114, 114, 114, 128, 142, 85], ]),
            '6': np.array([[99, 142, 57, 0, 0, 0, 0],
                           [99, 142, 91, 57, 57, 57, 34],
                           [99, 142, 91, 57, 99, 142, 85],
                           [99, 142, 125, 114, 128, 142, 85], ]),
            '7': np.array([[34, 114, 114, 114, 116, 142, 142],
                           [0, 0, 0, 0, 14, 142, 142],
                           [0, 0, 0, 0, 14, 142, 142],
                           [0, 0, 0, 0, 14, 142, 142], ]),
            '8': np.array([[99, 142, 125, 114, 128, 142, 85],
                           [99, 142, 91, 57, 99, 142, 85],
                           [99, 142, 91, 57, 99, 142, 85],
                           [99, 142, 125, 114, 128, 142, 85], ]),
            '9': np.array([[99, 142, 125, 114, 128, 142, 85],
                           [99, 142, 91, 57, 99, 142, 85],
                           [40, 57, 57, 57, 99, 142, 85],
                           [0, 0, 0, 0, 71, 142, 85], ])
        }
    elif which == 'hundreds':  # manorz: can't find an example with hundreds digit of order '9'. leave it for now.
        digits_filters = {
            '0': np.array([[142, 142, 114, 114, 139, 142, 28],
                           [142, 142, 0, 0, 128, 142, 28],
                           [142, 142, 0, 0, 128, 142, 28],
                           [142, 142, 114, 114, 139, 142, 28], ]),
            '1': np.array([[0, 0, 142, 142, 14, 0, 0],
                           [0, 0, 142, 142, 14, 0, 0],
                           [0, 0, 142, 142, 14, 0, 0],
                           [0, 0, 142, 142, 14, 0, 0], ]),
            '2': np.array([[114, 114, 114, 114, 139, 142, 28],
                           [57, 57, 57, 57, 133, 142, 28],
                           [142, 142, 57, 57, 57, 57, 11],
                           [142, 142, 114, 114, 114, 114, 23], ]),
            '3': np.array([[114, 114, 114, 114, 139, 142, 28],
                           [0, 0, 57, 57, 133, 142, 28],
                           [0, 0, 57, 57, 133, 142, 28],
                           [114, 114, 114, 114, 139, 142, 28], ]),
            '4': np.array([[142, 142, 0, 0, 128, 142, 28],
                           [142, 142, 57, 57, 133, 142, 28],
                           [57, 57, 57, 57, 133, 142, 28],
                           [0, 0, 0, 0, 128, 142, 28], ]),
            '5': np.array([[142, 142, 114, 114, 114, 114, 23],
                           [142, 142, 57, 57, 57, 57, 11],
                           [57, 57, 57, 57, 133, 142, 28],
                           [114, 114, 114, 114, 139, 142, 28], ]),
            '6': np.array([[142, 142, 0, 0, 0, 0, 0],
                           [142, 142, 57, 57, 57, 57, 11],
                           [142, 142, 57, 57, 133, 142, 28],
                           [142, 142, 114, 114, 139, 142, 28], ]),
            '7': np.array([[114, 114, 114, 114, 139, 142, 28],
                           [0, 0, 0, 0, 128, 142, 28],
                           [0, 0, 0, 0, 128, 142, 28],
                           [0, 0, 0, 0, 128, 142, 28], ]),
            '8': np.array([[142, 142, 114, 114, 139, 142, 28],
                           [142, 142, 57, 57, 133, 142, 28],
                           [142, 142, 57, 57, 133, 142, 28],
                           [142, 142, 114, 114, 139, 142, 28], ])
        }
    elif which == 'lives':  # manorz: 'zero' lives doesn't exist
        digits_filters = {
            '1': np.array([[0, 0, 57, 142, 99, 0, 0],
                           [0, 0, 57, 142, 99, 0, 0],
                           [0, 0, 57, 142, 99, 0, 0],
                           [0, 0, 57, 142, 99, 0, 0], ]),
            '2': np.array([[57, 114, 114, 114, 122, 142, 114],
                           [28, 57, 57, 57, 82, 142, 114],
                           [71, 142, 108, 57, 57, 57, 45],
                           [71, 142, 131, 114, 114, 114, 91], ]),
            '3': np.array([[57, 114, 114, 114, 122, 142, 114],
                           [0, 0, 23, 57, 82, 142, 114],
                           [0, 0, 23, 57, 82, 142, 114],
                           [57, 114, 114, 114, 122, 142, 114], ]),
            '4': np.array([[71, 142, 85, 0, 43, 142, 114],
                           [71, 142, 108, 57, 82, 142, 114],
                           [28, 57, 57, 57, 82, 142, 114],
                           [0, 0, 0, 0, 43, 142, 114], ]),
            '5': np.array([[71, 142, 131, 114, 114, 114, 91],
                           [71, 142, 108, 57, 57, 57, 45],
                           [28, 57, 57, 57, 82, 142, 114],
                           [57, 114, 114, 114, 122, 142, 114], ])
        }
    min_mse = np.inf
    digit = None
    for dig, filt in digits_filters.items():
        mse = ((obs - filt) ** 2).mean()
        if mse < min_mse:
            min_mse = mse
            digit = int(dig)
    return digit


def get_max_tunnel_depth(obs=None, method=cv2.TM_CCOEFF_NORMED):
    """
    return the avg. location of the padel over 4 consequtive observations
    obs: np.ndarray.shape = (3,74). obs is the buttom part of the image, where the padel is.
                                    we assume only the last frame in each observation as the
                                    relevant for the current location (the others are there to give the
                                    network a sense of motion).
    method: string in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                       cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
    """
    assert obs is not None
    assert obs.shape[0] == 14 and obs.shape[1] == 75
    all_depths = []
    for i in range(obs.shape[1]):
        depth_per_column = 0
        for j in reversed(range(obs.shape[0])):
            if obs[j, i] != 0:
                break
            else:
                depth_per_column += 1
        all_depths.append(depth_per_column)

    if len([x for x in all_depths if x != 0]) > 0:
        max_depth = max(all_depths)
    else:
        max_depth = 0

    #     print(all_depths)

    tunnel_open = max_depth == obs.shape[0]

    return max_depth, tunnel_open, all_depths


def get_padel_position(obs=None, method=cv2.TM_CCOEFF_NORMED):
    """
    return the avg. location of the padel over 4 consequtive observations
    obs: np.ndarray.shape = (3,74). obs is the buttom part of the image, where the padel is.
                                    we assume only the last frame in each observation as the
                                    relevant for the current location (the others are there to give the
                                    network a sense of motion).
    method: string in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                       cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
    """
    assert obs is not None
    assert obs.shape[0] == 3 and obs.shape[1] == 74
    assert obs.dtype in [np.uint8, np.float32]
    assert method in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                      cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    padel_filter = np.array(([44, 44, 44, 44, 44],
                             [110, 110, 110, 110, 110],
                             [22, 22, 22, 22, 22]), dtype=np.uint8)
    w, h = padel_filter.shape[::-1]

    res = cv2.matchTemplate(obs, padel_filter, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        center = min_loc + np.array([w / 2, h / 2])
    else:
        center = max_loc + np.array([w / 2, h / 2])

    center += [5, 75]
    print(max_loc, center)
    return center