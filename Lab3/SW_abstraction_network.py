class Signal_information(object):
    def __init__(self, power, path):
        self._signal_power = power
        self._noise_power = 0.0
        self._latency = 0.0
        self._path = path #[A, B, C, D, E, F]
    @property
    def signal_power(self):
        return self._signal_power
    @property
    def path(self):
        return self._path
    @path.setter
    def path(self, path):
        self._path = path
    @property
    def noise_power(self):
        return self._noise_power
    @noise_power.setter
    def noise_power(self, noise):
        self._noise_power = noise
    @property
    def latency(self):
        return self._latency
    @latency.setter
    def latency(self, latency):
        self._latency = latency
    def add_noise(self, noise):
        self.noise_power += noise
    def add_latency(self, latency):
        self.latency += latency
    def next(self):
        self.path = self.path [1:]
def main():
    #error

if __name__ == "__main__":
    main()
