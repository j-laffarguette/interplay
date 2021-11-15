import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from math import pi, sin
import cv2
import SimpleITK as sitk

mpl.use('Qt5Agg')


def positions(t0=0, tn=60, step=0.01, f=6, pn=100, to_plot=True):
    omega = 2 * pi * f  # Pulsation
    time = np.arange(t0, tn, omega * step)  # Time values for sin
    position = pn / 2 * np.sin(time + pi / 2) + pn / 2  # Positions along the time
    if to_plot:
        plt.plot(time, position)
        plt.xlabel('temps (s)')
        plt.show()

    return position, time


################################
# DEFINITIONS
################################

# Time values
t_tot = 25  # seconds
freq = 10
step = 0.0001

# sizes of itv and gtv
height = 250
ratio = 2
tot_length = height * ratio

s_itv = [height, height * ratio]
s_gtv = [height, height]
p0, pn = 0, s_itv[1] - s_itv[0]  # Available positions for gtv in itv (eg : from 0 to 100)
positions, time = positions(tn=t_tot, step=0.001, f=freq, pn=pn, to_plot=False)

# initialization of itv and gtv
itv = np.zeros(s_itv)
gtv = np.ones(s_gtv)

# Beams
num_beams = 5
b_size1, b_size2 = (s_itv[0], int(round(s_itv[1] / num_beams)))
beam = np.ones([b_size1, b_size2])
beam_positions = range(0,num_beams*b_size2,b_size2)


beam_position_in_time = []
for b in range(num_beams):
    beam_position_in_time += [beam_positions[b] for t in time if
                              int(t) in range(int(b * t_tot / num_beams), int((b + 1) * t_tot / num_beams))]

# dose matrix
dose = np.zeros(s_gtv)

# dose rate
d = 4  # Gy/min

if __name__ == "__main__":
    cv2.namedWindow("gtv")
    # cv2.namedWindow("dose")
    cv2.namedWindow("beam")

    for i, t, bp in zip(positions, time, beam_position_in_time):

        # New temporary area (size of itv)
        i = int(i)  # one needs integer values
        iitv = np.copy(itv)
        iitv_to_show = np.copy(itv)
        iitv[0:s_gtv[0], i:i + s_gtv[1]] = gtv
        iitv_to_show[0:s_gtv[0], i:i + s_gtv[1]] = dose / 255

        # intersection between iitv and beam
        temp_beam = np.zeros_like(iitv)
        temp_beam[0:s_itv[0], bp:bp + b_size2] = 1
        intersection = temp_beam[0:s_itv[0], i:i + s_gtv[1]]

        # dose accumulation
        dose = dose + intersection
        ind = np.unravel_index(np.argmax(dose, axis=None), dose.shape)

        # DOSE IMAGE
        # dose_to_display = dose
        # ind = np.unravel_index(np.argmax(dose, axis=None), dose.shape)
        # cv2.imwrite('dose.png', dose_to_display)
        # c = cv2.imread('dose.png')
        # gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        # cv2.copyMakeBorder(gray, 1000, 1000, 1000, 1000, cv2.BORDER_CONSTANT, None, value=0)
        # position of the max
        # gray_three = cv2.merge([gray, np.zeros_like(gray), gray])
        #
        # cv2.circle(gray_three, (int(ind[1]), int(s_gtv[0] / 2)), 3, (0, 0, 255), -1)
        # cv2.putText(gray_three, f"{np.max(dose)}", (int(ind[1]) + 10, int(s_gtv[0] / 2) + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 255, 255), 1)
        # cv2.imshow('dose', gray_three)

        cv2.rectangle(iitv_to_show, (i + s_gtv[1], s_gtv[0]), (i, 0), (255, 255, 0), 2)
        image = cv2.copyMakeBorder(iitv_to_show, 100, 100, 50, 50, cv2.BORDER_CONSTANT, None, value=0.4)
        cv2.imshow("gtv", image)

        # cv2.imshow('dose', dose/np.max(dose))

        cv2.imshow('beam', temp_beam)

        k = cv2.waitKey(int(20))
        if t == max(time):
            cv2.waitKey(0)
        if k == 27:
            break

