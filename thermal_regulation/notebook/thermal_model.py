import numpy as np


class ThermalModule():
    M = 0.19  # [kg]
    Q_solar = 300  # [Wm^-2]
    alpha_L = 0.936  # [-]
    C_l = 3762  # [JK^-1kg^-1]
    delta = 0.015  # [m]
    K_l = 0.502  # [WK^-1m^-1]
    h_L = 10.45  # [Wm^-2K^-1]
    a = 0.0314  # shape coeff
    A_L = 0.032  # a * np.pi * M **(2./3)  # [m^2]
    A_p = 0.4 * A_L  # [m^2]
    A_down = 0.3 * A_L  # [m^2]
    A_up = 0.6 * A_L  # [m^2]
    A_air = 0.9 * A_L  # [m^2]
    A_contact = 0.1 * A_L  # [m^2]
    eps_skin = 0.95  # [-]
    eps_land = 0.95  # [-]
    sigma = 5.67e-8  # Stefan-Boltzmann constant [Wm^-2K^-4]

    motor_coef = 1  # Motor-heat coefficient

    initial_T = 311.  # default. 38 C

    def __init__(self, temp_init=38.):
        """

        :param temp_init: Initial body temperature in Celsius degree
        """
        self.T = np.array(temp_init + 273.)
        self.initial_T = np.array(temp_init + 273.)

    def reset(self, temp_init=None):
        """

        :param temp_init: Initial body temperature in Celsius degree
        :return:
        """
        if temp_init:
            self.initial_T = np.array(temp_init + 273.)

        self.T = np.array(self.initial_T)

    def _dQ_solar(self):
        return self.alpha_L * self.A_p * self.Q_solar

    def _dQ_conv(self, temp_air):
        return self.h_L * self.A_air * (temp_air - self.T)

    def _dQ_longwave(self, temp_air, temp_earth):
        q_earth = self.eps_land * self.A_down * self.sigma * (temp_earth ** 4 - self.T ** 4)
        q_air = self.eps_skin * self.A_up * self.sigma * (temp_air ** 4 - self.T ** 4)
        return q_earth + q_air

    def _dQ_cond(self, temp_earth):
        return self.A_contact * self.K_l * (temp_earth - self.T) / (self.delta / 2.)

    def _delta_Q(self, action, evaporative_action, T_air, T_earth, is_shade=False):
        """
        Assuming all actions are scaled into [-1, +1]
        """

        dq1 = self._dQ_solar() * (not is_shade)  # solar ratiation
        dq2 = self._dQ_conv(T_air)  # convection heat
        dq3 = self._dQ_longwave(T_air, T_earth)  # long-wave heat
        dq4 = self._dQ_cond(T_earth)  # conductive heat
        dq5 = self.motor_coef * sum(np.square(action))  # motor heat production

        max_ev = 0.3
        min_ev = 0.272 * self.M
        dq6 = 0.5 * (max_ev - min_ev) * (evaporative_action + 1) + min_ev

        dQ = dq1 + dq2 + dq3 + dq4 + dq5 - dq6
        return dQ

    def _delta_T(self, action, evaporative_action, temp_air, temp_earth, is_shade):
        return self._delta_Q(action, evaporative_action, temp_air, temp_earth, is_shade) / (self.C_l * self.M)

    def step(self, action, evaporative_action, temp_air_c, temp_earth_c, is_shade, dt):
        """
        One-step progress of the thermal model. return the latest body temperature in Celsius degree
        :param action:
        :param evaporative_action:
        :param temp_air_c: air/sky temperature in Celsius degree
        :param temp_earth_c: earth/soil temperature in Celsius degree
        :param is_shade: is agent shaded or not
        :param dt: time tick of the thermal model
        :return:
        """
        self.T += self._delta_T(action, evaporative_action, temp_air_c + 273., temp_earth_c + 273., is_shade) * dt
        return self.T - 273.0


if __name__ == '__main__':

    # Testing function

    import matplotlib.pyplot as plt

    T_earth = 40    # [K]
    T_air = 35        # [K]
    T0_skin = 38.      # [K]

    dt = 0.05
    max_time = 60 * 30
    max_time_step = int(max_time / dt)

    def get_trajectory(ev_action):
        thermal_module = ThermalModule()
        thermal_module.reset()
        T_hist = []

        for i in range(max_time_step):

            # non Shade and in active
            is_shade = False
            action = 0.1 * np.random.uniform(-1, 1, 8)
            if i * dt > 500:  # Shade and Active
                is_shade = True
                action = np.random.uniform(-1, 1, 8)
            if i * dt > 1000: # Shade and in active
                is_shade = True
                action = 0.1 * np.random.uniform(-1, 1, 8)
            if i * dt > 1500: # not Shade and in active
                is_shade = False
                action = 0.5 * np.random.uniform(-1, 1, 8)

            eva_action = ev_action
            T = thermal_module.step(action, eva_action, T_air, T_earth, is_shade, dt)
            T_hist.append(T)
        return T_hist

    hist1 = get_trajectory(-1)
    plt.plot(np.linspace(0, max_time, max_time_step), hist1)

    hist2 = get_trajectory(+1)
    plt.plot(np.linspace(0, max_time, max_time_step), hist2)

    plt.legend(["minimum evapolation", "maximum evaporation"])
    plt.show()
