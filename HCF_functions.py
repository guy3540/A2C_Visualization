import numpy as np

import cv2

import sys


# Find the ball position in a Breakout-v4 gym environment
# Assume observation size of (210, 160)
# Note that output may be zeros if the ball was not found
def get_ball_position(obs=None):
    assert obs is not None
    assert (any([obs.shape[0] == 210, obs[1] == 160, obs[2] == 3]))

    area_X_start = 8
    area_X_end = 152
    area_Y_start = 32
    area_Y_end = 189
    min_allowed_cnt_area = 50
    img = obs[area_Y_start:area_Y_end, area_X_start:area_X_end].copy()
    # perform threshold on the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    et, thresh = cv2.threshold(gray, 0, 255, 1)
    pad_size = 20
    thresh = np.pad(thresh, (pad_size, pad_size), 'constant', constant_values=(255, 255))
    contours, h = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_cnt_area = sys.maxsize  # Initialize a min variable
    center = np.array([0, 0])
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        # if there's a containig ellipse to the contour, we'll extract its center
        cnt_area = cv2.contourArea(cnt)
        if all([len(approx) <= 8, cnt_area < min_cnt_area, cnt_area < min_allowed_cnt_area]):
            min_cnt_area = cnt_area
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) - pad_size  # X center
            cY = int(M["m01"] / M["m00"]) - pad_size  # Y center
            # print("square", cX, cY)
            cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)
            center = np.array([cY + area_Y_start, cX + area_X_start])
    return center.tolist()
    # return max_val, loc, area


def get_max_tunnel_depth(obs=None, method=cv2.TM_CCOEFF_NORMED):
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
        return [max_depth, tunnel_open, all_depths.tolist()]
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
    return [max_depth, tunnel_open, all_depths.tolist()]


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
        # if there's a containig ellipse to the contour, we'll extract its center
        if all([len(approx) >= 8, cv2.contourArea(cnt) > max_cnt_area]):
            max_cnt_area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) - pad_size  # X center
            cY = int(M["m01"] / M["m00"]) - pad_size  # Y center
            # print("square", cX, cY)
            cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)
            center = np.array([cY+area_Y_start, cX+area_X_start])
    return center.tolist()


# I tried to implement the get_digit function in a way that extracts the number from each observation.
# Unfortunately, that doesn't seem to work (segmentation is perfect, but pytesseract fails to accurately
# detect the digits).
# Therefore I decided to collect the score by overriding the step function in the gym wrapper

# def get_digit(obs=None, which=None, method=cv2.TM_CCOEFF_NORMED):
#     assert (any([obs.shape[0] == 210, obs[1] == 160, obs[2] == 3]))
#     area_Y_start = 0
#     area_Y_end = 17
#     score_area_start = 30
#     score_area_end = 90
#     lives_area_start = 90
#     lives_area_end = 120
#     pad_size = 5
#
#     numbers_area = obs[area_Y_start:area_Y_end, :]
#     gray = cv2.cvtColor(numbers_area, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     score_im = thresh[:, score_area_start:score_area_end]
#     lives_im = thresh[:, lives_area_start:lives_area_end]
#
#     score = 0
#     cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
#     print("contours size is: " + str(len(cnts)))
#     img = cv2.drawContours(numbers_area, cnts, -1, (0, 255, 0), 3)
#
#     plot_rows = 4
#     ax1 = plt.subplot2grid((plot_rows, len(cnts)), (0, 0), rowspan=1, colspan=len(cnts))
#     ax1.imshow(thresh)
#     ax1.set_title('thresh')
#     ax2 = plt.subplot2grid((plot_rows, len(cnts)), (1, 0), rowspan=1, colspan=len(cnts))
#     ax2.imshow(img)
#     ax2.set_title('all contours')
#
#     for c, num in zip(cnts, range(len(cnts))):
#         print(num)
#         x, y, w, h = cv2.boundingRect(c)
#         ROI = 255 - thresh[y:y + h, x:x + w]
#         ROI = np.pad(ROI, (pad_size, pad_size), 'constant', constant_values=(255, 255))
#         filename = 'ROI_{}.png'.format(num)
#         cv2.imwrite(filename, ROI)
#         ax = plt.subplot2grid((plot_rows, len(cnts)), (2, num), rowspan=1, colspan=1)
#         ax.imshow(ROI)
#
#         current_digit = pytesseract.image_to_string(cv2.imread(filename, 0), lang='eng',
#                                                     config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')  #  -c tessedit_char_whitelist=0123456789al
#         current_digit.replace('a', '0')
#         current_digit.replace('l', '1')
#         current_digit.replace('z', '2')
#         ax.set_title(current_digit)
#         os.remove(filename)
#
#         cv2.imwrite(filename, 255 - ROI)
#
#         current_digit = pytesseract.image_to_string(cv2.imread(filename, 0), lang='eng',
#                                                     config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')  # -c tessedit_char_whitelist=0123456789al
#         ax = plt.subplot2grid((plot_rows, len(cnts)), (3, num), rowspan=1, colspan=1)
#         ax.imshow(255 - ROI)
#         ax.set_title(current_digit)
#         os.remove(filename)
#     plt.tight_layout()
#     plt.show()
#     print("score is: " + str(score))


#     The following code is the old function, which assumes downsampling of the observation.
#     The new version, however, assumes that the function is run before the downsampling.
#     Threrfore we were able to provide a nicer, more readable code


#     """
#     return the digit from raw image
#     method: string in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
#                        cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
#     """
#     assert obs is not None
#     assert obs.shape[0] == 4 and obs.shape[1] == 7
#     assert obs.dtype in [np.uint8, np.float32]
#     assert method in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
#                       cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
#     assert which in ['hundreds', 'tens', 'ones', 'lives']
#
#     if which == 'ones':  # units digit
#         digits_filters = {
#             '0': np.array([[43, 142, 136, 114, 116, 142, 142],
#                            [43, 142, 114, 0, 14, 142, 142],
#                            [43, 142, 114, 0, 14, 142, 142],
#                            [43, 142, 136, 114, 116, 142, 142], ]),
#             '1': np.array([[0, 0, 28, 142, 128, 0, 0],
#                            [0, 0, 28, 142, 128, 0, 0],
#                            [0, 0, 28, 142, 128, 0, 0],
#                            [0, 0, 28, 142, 128, 0, 0], ]),
#             '2': np.array([[34, 114, 114, 114, 116, 142, 142],
#                            [17, 57, 57, 57, 65, 142, 142],
#                            [43, 142, 125, 57, 57, 57, 57],
#                            [43, 142, 136, 114, 114, 114, 114], ]),
#             '3': np.array([[34, 114, 114, 114, 116, 142, 142],
#                            [0, 0, 11, 57, 65, 142, 142],
#                            [0, 0, 11, 57, 65, 142, 142],
#                            [34, 114, 114, 114, 116, 142, 142], ]),
#             '4': np.array([[43, 142, 114, 0, 14, 142, 142],
#                            [43, 142, 125, 57, 65, 142, 142],
#                            [17, 57, 57, 57, 65, 142, 142],
#                            [0, 0, 0, 0, 14, 142, 142], ]),
#             '5': np.array([[43, 142, 136, 114, 114, 114, 114],
#                            [43, 142, 125, 57, 57, 57, 57],
#                            [17, 57, 57, 57, 65, 142, 142],
#                            [34, 114, 114, 114, 116, 142, 142], ]),
#             '6': np.array([[43, 142, 114, 0, 0, 0, 0],
#                            [43, 142, 125, 57, 57, 57, 57],
#                            [43, 142, 125, 57, 65, 142, 142],
#                            [43, 142, 136, 114, 116, 142, 142], ]),
#             '7': np.array([[34, 114, 114, 114, 116, 142, 142],
#                            [0, 0, 0, 0, 14, 142, 142],
#                            [0, 0, 0, 0, 14, 142, 142],
#                            [0, 0, 0, 0, 14, 142, 142], ]),
#             '8': np.array([[43, 142, 136, 114, 116, 142, 142],
#                            [43, 142, 125, 57, 65, 142, 142],
#                            [43, 142, 125, 57, 65, 142, 142],
#                            [43, 142, 136, 114, 116, 142, 142], ]),
#             '9': np.array([[43, 142, 136, 114, 116, 142, 142],
#                            [43, 142, 125, 57, 65, 142, 142],
#                            [17, 57, 57, 57, 65, 142, 142],
#                            [0, 0, 0, 0, 14, 142, 142], ])
#         }
#     elif which == 'tens':  # tens digit
#         digits_filters = {
#             '0': np.array([[99, 142, 125, 114, 128, 142, 85],
#                            [99, 142, 57, 0, 71, 142, 85],
#                            [99, 142, 57, 0, 71, 142, 85],
#                            [99, 142, 125, 114, 128, 142, 85], ]),
#             '1': np.array([[0, 0, 85, 142, 71, 0, 0],
#                            [0, 0, 85, 142, 71, 0, 0],
#                            [0, 0, 85, 142, 71, 0, 0],
#                            [0, 0, 85, 142, 71, 0, 0], ]),
#             '2': np.array([[80, 114, 114, 114, 128, 142, 85],
#                            [40, 57, 57, 57, 99, 142, 85],
#                            [99, 142, 91, 57, 57, 57, 34],
#                            [99, 142, 125, 114, 114, 114, 68], ]),
#             '3': np.array([[80, 114, 114, 114, 128, 142, 85],
#                            [0, 0, 34, 57, 99, 142, 85],
#                            [0, 0, 34, 57, 99, 142, 85],
#                            [80, 114, 114, 114, 128, 142, 85], ]),
#             '4': np.array([[99, 142, 57, 0, 71, 142, 85],
#                            [99, 142, 91, 57, 99, 142, 85],
#                            [40, 57, 57, 57, 99, 142, 85],
#                            [0, 0, 0, 0, 71, 142, 85], ]),
#             '5': np.array([[99, 142, 125, 114, 114, 114, 68],
#                            [99, 142, 91, 57, 57, 57, 34],
#                            [40, 57, 57, 57, 99, 142, 85],
#                            [80, 114, 114, 114, 128, 142, 85], ]),
#             '6': np.array([[99, 142, 57, 0, 0, 0, 0],
#                            [99, 142, 91, 57, 57, 57, 34],
#                            [99, 142, 91, 57, 99, 142, 85],
#                            [99, 142, 125, 114, 128, 142, 85], ]),
#             '7': np.array([[34, 114, 114, 114, 116, 142, 142],
#                            [0, 0, 0, 0, 14, 142, 142],
#                            [0, 0, 0, 0, 14, 142, 142],
#                            [0, 0, 0, 0, 14, 142, 142], ]),
#             '8': np.array([[99, 142, 125, 114, 128, 142, 85],
#                            [99, 142, 91, 57, 99, 142, 85],
#                            [99, 142, 91, 57, 99, 142, 85],
#                            [99, 142, 125, 114, 128, 142, 85], ]),
#             '9': np.array([[99, 142, 125, 114, 128, 142, 85],
#                            [99, 142, 91, 57, 99, 142, 85],
#                            [40, 57, 57, 57, 99, 142, 85],
#                            [0, 0, 0, 0, 71, 142, 85], ])
#         }
#     elif which == 'hundreds':  # manorz: can't find an example with hundreds digit of order '9'. leave it for now.
#         digits_filters = {
#             '0': np.array([[142, 142, 114, 114, 139, 142, 28],
#                            [142, 142, 0, 0, 128, 142, 28],
#                            [142, 142, 0, 0, 128, 142, 28],
#                            [142, 142, 114, 114, 139, 142, 28], ]),
#             '1': np.array([[0, 0, 142, 142, 14, 0, 0],
#                            [0, 0, 142, 142, 14, 0, 0],
#                            [0, 0, 142, 142, 14, 0, 0],
#                            [0, 0, 142, 142, 14, 0, 0], ]),
#             '2': np.array([[114, 114, 114, 114, 139, 142, 28],
#                            [57, 57, 57, 57, 133, 142, 28],
#                            [142, 142, 57, 57, 57, 57, 11],
#                            [142, 142, 114, 114, 114, 114, 23], ]),
#             '3': np.array([[114, 114, 114, 114, 139, 142, 28],
#                            [0, 0, 57, 57, 133, 142, 28],
#                            [0, 0, 57, 57, 133, 142, 28],
#                            [114, 114, 114, 114, 139, 142, 28], ]),
#             '4': np.array([[142, 142, 0, 0, 128, 142, 28],
#                            [142, 142, 57, 57, 133, 142, 28],
#                            [57, 57, 57, 57, 133, 142, 28],
#                            [0, 0, 0, 0, 128, 142, 28], ]),
#             '5': np.array([[142, 142, 114, 114, 114, 114, 23],
#                            [142, 142, 57, 57, 57, 57, 11],
#                            [57, 57, 57, 57, 133, 142, 28],
#                            [114, 114, 114, 114, 139, 142, 28], ]),
#             '6': np.array([[142, 142, 0, 0, 0, 0, 0],
#                            [142, 142, 57, 57, 57, 57, 11],
#                            [142, 142, 57, 57, 133, 142, 28],
#                            [142, 142, 114, 114, 139, 142, 28], ]),
#             '7': np.array([[114, 114, 114, 114, 139, 142, 28],
#                            [0, 0, 0, 0, 128, 142, 28],
#                            [0, 0, 0, 0, 128, 142, 28],
#                            [0, 0, 0, 0, 128, 142, 28], ]),
#             '8': np.array([[142, 142, 114, 114, 139, 142, 28],
#                            [142, 142, 57, 57, 133, 142, 28],
#                            [142, 142, 57, 57, 133, 142, 28],
#                            [142, 142, 114, 114, 139, 142, 28], ])
#         }
#     elif which == 'lives':  # manorz: 'zero' lives doesn't exist
#         digits_filters = {
#             '1': np.array([[0, 0, 57, 142, 99, 0, 0],
#                            [0, 0, 57, 142, 99, 0, 0],
#                            [0, 0, 57, 142, 99, 0, 0],
#                            [0, 0, 57, 142, 99, 0, 0], ]),
#             '2': np.array([[57, 114, 114, 114, 122, 142, 114],
#                            [28, 57, 57, 57, 82, 142, 114],
#                            [71, 142, 108, 57, 57, 57, 45],
#                            [71, 142, 131, 114, 114, 114, 91], ]),
#             '3': np.array([[57, 114, 114, 114, 122, 142, 114],
#                            [0, 0, 23, 57, 82, 142, 114],
#                            [0, 0, 23, 57, 82, 142, 114],
#                            [57, 114, 114, 114, 122, 142, 114], ]),
#             '4': np.array([[71, 142, 85, 0, 43, 142, 114],
#                            [71, 142, 108, 57, 82, 142, 114],
#                            [28, 57, 57, 57, 82, 142, 114],
#                            [0, 0, 0, 0, 43, 142, 114], ]),
#             '5': np.array([[71, 142, 131, 114, 114, 114, 91],
#                            [71, 142, 108, 57, 57, 57, 45],
#                            [28, 57, 57, 57, 82, 142, 114],
#                            [57, 114, 114, 114, 122, 142, 114], ])
#         }
#     min_mse = np.inf
#     digit = None
#     for dig, filt in digits_filters.items():
#         mse = ((obs - filt) ** 2).mean()
#         if mse < min_mse:
#             min_mse = mse
#             digit = int(dig)
#     return digit


# Analyze the bricks area in a Breakout-v4 gym environment
# Assume observation size of (210, 160)