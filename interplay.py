import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from math import pi, sin
import cv2
import SimpleITK as sitk


class Dose:
    def __init__(self, num_beams, bpm, dose_waited, dose_rate):
        # sizes of itv and gtv
        self.height = 250
        self.ratio = 3
        self.tot_length = self.height * self.ratio

        # initialization of itv and gtv
        self.s_itv = [self.height, self.height * self.ratio]
        self.s_gtv = [self.height, self.height]
        self.itv = np.zeros(self.s_itv)
        self.gtv = np.ones(self.s_gtv)

        # Time values
        self.freq = bpm/60  # Hz
        self.tn = 100  # End of calculation (s) - To be sure that one have at least 60 min of beam
        self.step = 0.005  # the step of calculation is 1 ms

        self.p0, self.pn = 0, self.s_itv[1] - self.s_itv[0]  # Available positions for gtv in itv (eg : from 0 to 100)
        self.positions = []
        self.time = []
        self.get_positions()  # creates an array containing all the positions of gtv and all time increments

        # dose matrix
        self.dose = np.zeros(self.s_gtv)
        self.dose_rate = dose_rate
        self.dose_needed = dose_waited

        # Beams
        self.num_beams = num_beams
        self.b_size1, self.b_size2 = (self.s_itv[0], int(round(self.s_itv[1] / self.num_beams)))
        self.beam_position_in_time = []
        self.time_needed = 0
        self.do_beams()

    def get_positions(self, sinus = True, to_plot=True):
        t0 = 0
        omega = 2 * pi * self.freq  # Pulsation
        self.time = np.arange(t0, self.tn, self.step)  # Time values for sin

        if sinus:
            self.positions = self.pn / 2 * np.sin(omega * self.time-pi/2) + self.pn / 2  # Positions along the time
        else:
            self.positions = self.pn* np.sin(omega * self.time/2)**4

        if to_plot:

            plt.plot(self.time[0:1000], self.positions[0:1000])
            plt.xlabel('temps (s)')
            plt.show()

    def do_beams(self):
        self.time_needed = self.dose_needed / self.dose_rate * 60
        beam = np.ones([self.b_size1, self.b_size2])
        beam_positions = range(0, self.num_beams * self.b_size2, self.b_size2)

        for b in range(self.num_beams):
            self.beam_position_in_time += [beam_positions[b] for t in self.time if
                                           int(t) in range(int(b * self.time_needed), int(self.time_needed * (b + 1)))]
        print()

    def main_loop(self):
        index = 0
        for i, t, bp in zip(self.positions, self.time, self.beam_position_in_time):

            index += 1

            # New temporary area (size of itv)
            i = int(i)  # one needs integer values
            iitv = np.copy(self.itv)
            iitv_to_show = np.copy(self.itv)
            iitv[0:self.s_gtv[0], i:i + self.s_gtv[1]] = self.gtv
            iitv_to_show[0:self.s_gtv[0], i:i + self.s_gtv[1]] = self.dose / 255

            # intersection between iitv and beam
            temp_beam = np.zeros_like(iitv)
            temp_beam[0:self.s_itv[0], bp:bp + self.b_size2] = 1
            intersection = temp_beam[0:self.s_itv[0], i:i + self.s_gtv[1]]

            # dose accumulation
            self.dose = self.dose + intersection
            ind = np.unravel_index(np.argmax(self.dose, axis=None), self.dose.shape)

            cv2.rectangle(iitv_to_show, (i + self.s_gtv[1], self.s_gtv[0]), (i, 0), (255, 255, 0), 2)
            image = cv2.copyMakeBorder(iitv_to_show, 100, 100, 50, 50, cv2.BORDER_CONSTANT, None, value=0.4)
            cv2.imshow("gtv", image)

            # cv2.imshow('dose', dose/np.max(dose))

            if not(t*1000)%5:
                cv2.imshow('beam', temp_beam)
                k = cv2.waitKey(int(1))
                if index == len(self.beam_position_in_time):
                    cv2.waitKey(0)
                if k == 27:
                    break


if __name__ == "__main__":
    num_beams = 8
    bpm = 12
    dose_waited = 0.5
    dose_rate = 20

    d = Dose(num_beams, bpm, dose_waited, dose_rate)
    d.main_loop()

    # cv2.namedWindow("gtv")
    # # cv2.namedWindow("dose")
    # cv2.namedWindow("beam")
    # #
    # for i, t, bp in zip(positions, time, beam_position_in_time):
    #
    #     # New temporary area (size of itv)
    #     i = int(i)  # one needs integer values
    #     iitv = np.copy(itv)
    #     iitv_to_show = np.copy(itv)
    #     iitv[0:s_gtv[0], i:i + s_gtv[1]] = gtv
    #     iitv_to_show[0:s_gtv[0], i:i + s_gtv[1]] = dose / 255
    #
    #     # intersection between iitv and beam
    #     temp_beam = np.zeros_like(iitv)
    #     temp_beam[0:s_itv[0], bp:bp + b_size2] = 1
    #     intersection = temp_beam[0:s_itv[0], i:i + s_gtv[1]]
    #
    #     # dose accumulation
    #     dose = dose + intersection
    #     ind = np.unravel_index(np.argmax(dose, axis=None), dose.shape)
    #
    #     # DOSE IMAGE
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
    #
    #
