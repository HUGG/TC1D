"""
Calculates fission track length distribution

Description:
    MadTrax is a Fortran program originally from Braun et al. (2006). This is
    a Python version of it and works similarly than the original one. This program
    calculates the age of a sample and its length distribution.
    I added also a plot for the distribution.
    
Author:
    Lotta Ylä-Mella 15.1.2018

"""

import numpy as np
import matplotlib.pyplot as plt


def g(r, a, b):
    g = (((1 - r**b) / b) ** a - 1) / a
    return g


def xinv(gr, a, b):
    xinv = (1.0 - b * (a * gr + 1) ** (1 / a)) ** (1 / b)
    return xinv


def temperature(temp, time, t, n):
    temperature = temp[0]

    for i in range(0, n - 1):
        if ((t - time[i])) * (t - time[i + 1]) <= 0.0:
            rat = (t - time[i]) / (time[i + 1] - time[i])
            temperature = temp[i] + rat * (temp[i + 1] - temp[i])
            return temperature

    temperature = temp[n - 1]
    return temperature


def xk(x):
    xk = 0.39894228 * np.exp(-(x**2) / 2.0)
    return xk


# INPUT
# time_i:the time values (in Myr) in descending order at which the thermal history
# is given (ex: 100,50,20,10,0); the last value should always be 0; the first value
# should be smaller than 1000.
# temp_i: the thermal history in degree Celsius
# n: the number of time-temperature pairs used  to describe the temperature history
# out_flag:  =0 only calculate fission track age
#            =1 also calculate track length distribution and statistics
# param_flag : =1 uses Laslett et al, 1987 parameters
#              =2 uses Crowley et al., 1991 Durango parameters
#              =3 uses Crowley et al., 1991 F-apatite parameters


def madtrax_apatite(
    time_i, temp_i, n, out_flag, param_flag
):  # , adv, timeMa): #, fta, ftld, ftldmean, ftldsd):
    r = np.zeros(1000)
    prob = np.zeros(101)
    ftld = np.zeros(17)

    if param_flag == 1:
        # Laslett et al. 1987
        a = 0.35
        b = 2.7
        c0 = -4.87
        c1 = 0.000168
        c2 = 0.00472416
        c3 = 0.0
    elif param_flag == 2:
        # Crowley et al., 1991 (Durango)
        a = 0.49
        b = 3.0
        c0 = -3.202
        c1 = 0.0000937
        c2 = 0.001839
        c3 = 0.00042
    else:
        # Crowley et al., 1991 (F-apatite)
        a = 0.76
        b = 4.16
        c0 = -1.508
        c1 = 0.00002076
        c2 = 0.0002143
        c3 = 0.0009967

    # unannealed fission track length
    xind = 16.0

    # mean length of spontaneous tracks in standards
    xfct = 14.5

    # calculate the number of time steps assuming 1My time step length
    # if run time > 100 My, else take 100 time steps
    nstep = int(time_i[0])

    # if (nstep.gt.8000) stop 'Your final time is greater than '//
    # &                        'the age of the Universe...'
    # if (nstep.gt.4500) stop 'Your final time is greater than '//
    # &                        'the age of the Earth...'
    # if (nstep.gt.1000) stop 'Fission track does not work very well '//
    # &                        'for time spans greater than 1Byr...'
    if nstep > 1000:
        return
    time_interval = 1.0
    if nstep < 100:
        nstep = 100
        time_interval = time_i[0] / 100.0
    deltat = time_interval * 1e6 * 365.24 * 24 * 3600

    # calculate final temperature
    tempp = temperature(temp_i, time_i, 0.0, n) + 273.0
    rp = 0.5

    i = 0
    # begining of time stepping
    while i < nstep:  # 0, nstep):
        time = (i + 1) * time_interval
        # calculate temperature by linear interpolation
        temp = temperature(temp_i, time_i, time, n) + 273.0
        # calculate mean temperature over the time step
        tempm = (temp + tempp) / 2.0
        # calculate the "equivalent time", teq
        teq = np.exp((-c2 / c1) + ((g(rp, a, b) - c0) / c1) * (1.0 / tempm - c3))

        if i == 0:
            teq = 0

        # check if we are not getting too close to r=0
        # in which case r remains 0 for all following time steps
        if (
            np.log(teq + deltat)
            >= ((1 / b) ** a - a * c0 - 1.0) / a / c1 * (1.00 / tempm - c3) - c2 / c1
        ):
            for j in range(i, nstep):
                r[j] = 0.0
            nstep = i + 1
        else:

            # otherwise calculte reduction in length, r, over the time step, dt
            dt = teq + deltat
            gr = c0 + ((c1 * np.log(dt) + c2) / ((1.0 / tempm) - c3))
            r[i] = xinv(gr, a, b)

            # update variables for next time step
            tempp = temp
            rp = r[i]
        i = i + 1

    # all reduction factors for all time steps have been calculated
    # now estimate the fission track age by simple summation
    # (here it helps to use 1 Myr time steps)
    sumdj = 0.0

    for i in range(0, nstep):
        if r[i] <= 0.35:
            dj = 0
        elif r[i] <= 0.66:
            dj = 2.15 * r[i] - 0.76
        else:
            dj = r[i]

        sumdj = sumdj + dj

    fta = (xind / xfct) * sumdj * time_interval

    # now if out_flag is not equal to 0 let's do some statistics
    if out_flag == 0.0:
        print(fta)
        return fta

    sumprob = 0.0

    # Probabilities
    for j in range(0, 101):
        rr = j / 100.0

        if rr <= 0.43:
            h = 2.53
        elif rr <= 0.67:
            h = 5.08 - 5.93 * rr
        else:
            h = 1.39 - 0.61 * rr

        fr = 0

        for i in range(nstep):
            x = (rr - r[i]) * xind / h
            dfr = xk(x) / h
            fr = fr + dfr

        prob[j] = fr / nstep
        sumprob = sumprob + prob[j]

    # now let's rescale the track length distribution, its mean and standard
    # deviation

    ftld[16] = 100.0
    imin = 1

    # distribution
    for l in range(0, 16):
        imax = int((l + 1) * 100 / xind)
        ftld[l] = 0.0

        for i in range(imin, imax + 1):
            ftld[l] = ftld[l] + prob[i - 1]

        ftld[l] = ftld[l] * 100.0 / sumprob
        ftld[16] = ftld[16] - ftld[l]

        imin = imax + 1

    # mean length
    sumftld = 0.0

    for l in range(0, 17):
        sumftld = sumftld + ftld[l] * ((l + 1) - 0.5)

    ftldmean = sumftld / 100.0

    # standard devitation
    devftld = 0.0

    for l in range(0, 17):
        devftld = devftld + ftld[l] * ((l + 1) - 0.5 - ftldmean) ** 2

    ftldsd = np.sqrt(devftld / 100.0)

    # parameters for a figure
    N = len(ftld)  # number of lengths
    ind = np.arange(N)  # sequence of scalars
    width = 1  # width of the bar

    # fig, ax = plt.subplots()
    # ax.bar(ind, ftld, width,color='b', edgecolor='k')
    # ax.bar(ind-0.5, ftld, width)

    # font = 12
    # ax.set_xlabel('Length ($\mu$m)', fontsize=font)
    # ax.set_ylabel('Relative frequency (%)', fontsize=font)
    # ax.annotate('Total time: ' + str(timeMa) + ' Ma', xy=(-4,27), fontsize=font) #19
    # ax.annotate('Advection velocity: ' + str(adv) + ' mm/a', xy=(-4,24), fontsize=font) #17
    # ax.annotate('Age: %.0f Ma' %fta,xy=(-4,21), fontsize=font) #15

    # Different axis to plot a figure
    # ax.axis([-5, 20, 0, 25])
    # ax.axis([-5,20,0,35])

    # plt.savefig(str(adv) + '.png')
    # plt.show()

    # Collection of all results (age, final length, mean length, standard deviation)
    result = [fta * 10**6, ftld[16], ftldmean, ftldsd]

    return result
