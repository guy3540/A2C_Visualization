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

# very slow function
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
    assert (any([obs.shape[0] == 210, obs[1] == 160, obs[2] == 3]))
    # assert obs.shape[0] == 14 and obs.shape[1] == 75
    brick_area = obs[57:93, 8:-8]
    gray = cv2.cvtColor(brick_area, cv2.COLOR_BGR2GRAY)
    et, thresh = cv2.threshold(gray, 50, 255, 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:    # contours is empty - no bricks broken at all
        max_depth = 0
        tunnel_open = False
        all_depths = np.zeros(brick_area.shape[1])
        return max_depth, tunnel_open, all_depths
    h_max = 0
    longest_col = contours[0]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > h_max:
            longest_col = cnt
    x, y, w, h = cv2.boundingRect(cnt)
    max_depth = h
    tunnel_open = (h >= brick_area.shape[0])
    all_depths = thresh.sum(axis=0) / 255
    return max_depth, tunnel_open, all_depths


# Find the paddle position in a Breakout-v4 gym environment
# Assume observation size of (210, 160)
def get_paddle_position(obs=None, method=cv2.TM_CCOEFF_NORMED):
    assert(any([obs.shape[0] == 210, obs[1] == 160, obs[2] == 3]))
    # paddle is always found in the lower part of the observation
    # set desired area to look for paddle
    area_X_start = 8
    area_X_end = 152
    area_Y_start = 185
    area_Y_end = 195
    img = obs[area_Y_start:area_Y_end, area_X_start:area_X_end].copy()
    # perform threshold on the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    et, thresh = cv2.threshold(gray, 50, 255, 1)
    pad_size = 20
    thresh = np.pad(thresh, (pad_size, pad_size), 'constant', constant_values=(255, 255))
    contours, h = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt_area = 0
    center = np.array([(area_Y_start+area_Y_end)/2, (area_X_start+area_X_end)/2])
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        # print("contour", cnt.shape)
        # if there's a containig ellipse to the contour, we'll extract its center
        if all([len(approx) >= 8, cv2.contourArea(cnt) > max_cnt_area]):
            max_cnt_area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) - pad_size  # X center
            cY = int(M["m01"] / M["m00"]) - pad_size  # Y center
            # print("square", cX, cY)
            cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)
            center = np.array([cY+area_Y_start, cX+area_X_start])
    return center
