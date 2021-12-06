from time import sleep
from matplotlib import pyplot as plt
import numpy as np

from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf, read_raw_eeglab
import mne

delta_min, delta_max, theta_min, theta_max, alpha_min, alpha_max, beta_min, beta_max, gamma_min, gamma_max = 0, 4, 4, 8, 8, 13, 13, 30, 30, 40
fmin, fmax = 2, 40

def fr_band(x, y, min, max):
    a = 0
    b = len(x)
    for i in range(len(x)):
        if x[i] >= min:
            a = i
            break
    for i in range(len(x) - 1, 0, -1):
        if x[i] <= max:
            b = i + 1
            break

    return x[a:b], y[a:b]


def rhythms_rel_power(psd, channel):
    x, y = psd[1], psd[0][channel]
    delta_x, delta_y = fr_band(x, y, delta_min, delta_max)
    theta_x, theta_y = fr_band(x, y, theta_min, theta_max)
    alpha_x, alpha_y = fr_band(x, y, alpha_min, alpha_max)
    beta_x, beta_y = fr_band(x, y, beta_min, beta_max)
    gamma_x, gamma_y = fr_band(x, y, gamma_min, gamma_max)

    total_power = np.trapz(y, x)
    delta_power = np.trapz(delta_y, delta_x)
    theta_power = np.trapz(theta_y, theta_x)
    alpha_power = np.trapz(alpha_y, alpha_x)
    beta_power = np.trapz(beta_y, beta_x)
    gamma_power = np.trapz(gamma_y, gamma_x)

    rel_delta_power = delta_power / total_power
    rel_theta_power = theta_power / total_power
    rel_alpha_power = alpha_power / total_power
    rel_beta_power = beta_power / total_power
    rel_gamma_power = gamma_power / total_power

    return rel_delta_power, rel_theta_power, rel_alpha_power, rel_beta_power, rel_gamma_power

raw = read_raw_edf("testeeg.edf", eog=(), preload=True)
raw.describe()

print(raw.info["ch_names"])

for channel in range(64):
    time_scale = []
    rel_delta_power_values = []
    rel_theta_power_values = []
    rel_alpha_power_values = []
    rel_beta_power_values = []
    rel_gamma_power_values = []

    d = 5
    tmax = 140
    for i in range(0, tmax, d):
        psd = mne.time_frequency.psd_multitaper(raw, fmin=fmin, fmax=fmax, tmin=i, tmax=i+d)
        rel_delta_power, rel_theta_power, rel_alpha_power, rel_beta_power, rel_gamma_power = rhythms_rel_power(psd, channel)
        time_scale.append(i + d/2)
        rel_delta_power_values.append(rel_delta_power)
        rel_theta_power_values.append(rel_theta_power)
        rel_alpha_power_values.append(rel_alpha_power)
        rel_beta_power_values.append(rel_beta_power)
        rel_gamma_power_values.append(rel_gamma_power)


    plt.plot(time_scale, rel_delta_power_values, label="delta")
    plt.plot(time_scale, rel_theta_power_values, label="theta")
    plt.plot(time_scale, rel_alpha_power_values, label="alpha")
    plt.plot(time_scale, rel_beta_power_values, label="beta")
    plt.plot(time_scale, rel_gamma_power_values, label="gamma")
    plt.title(raw.info["ch_names"][channel])
    plt.legend(loc="upper left")
    plt.show()


    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size

    plt.plot(time_scale, np.convolve(rel_delta_power_values, kernel, mode='same'), label="delta")
    plt.plot(time_scale, np.convolve(rel_theta_power_values, kernel, mode='same'), label="theta")
    plt.plot(time_scale, np.convolve(rel_alpha_power_values, kernel, mode='same'), label="alpha")
    plt.plot(time_scale, np.convolve(rel_beta_power_values, kernel, mode='same'), label="beta")
    plt.plot(time_scale, np.convolve(rel_gamma_power_values, kernel, mode='same'), label="gamma")
    plt.title(raw.info["ch_names"][channel])
    plt.legend(loc="upper left")
    plt.show()




# midline = ['EEG CP1-CPz']
# raw.plot_psd(picks=midline, fmax=50, average=True, tmin=600, tmax=605)
# psd = mne.time_frequency.psd_multitaper(raw, fmin=fmin, fmax=fmax, tmin=600, tmax=605)
# print(rhythms_rel_power(psd, channel))



