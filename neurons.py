import json
from multiprocessing import Array, Process, Value
from time import sleep, time

import numpy as np
import zmq


class NeuronStream(Process):
    
    def __init__(self, channels=16, buffer_size=128+16, dt=16/240):
        super(Process, self).__init__()
        
        self.channels = channels
        self.buffer_size = buffer_size
        self.dt = dt
        
        self.run_loop = Value("i", 1)
        self.enough_time = Value("i", 0)
        
        self.spike_history_len = 20
        self.raw_spikes_buffer = Array("d", [0] * self.spike_history_len * channels)
        self.spike_offsets = Array("i", [0] * channels)
        self.spike_frequency_buffer = Array("d", [0] * self.buffer_size * channels)
        self.spike_frequency_offset = Value("i", 0)
        
        self.last_spike_readout = None
        
    def send_heartbeat(self):
        d = {'application': "Biobot",
             'uuid': "uuid",
             'type': 'heartbeat'}
        j_msg = json.dumps(d)
        self.heartbeat_socket.send(j_msg.encode('utf-8'))
        
        self.last_heartbeat = time()
        self.socket_waits_reply = True
        
    def check_heartbeat(self):
        if (time() - self.last_heartbeat) > 3.:
            if self.socket_waits_reply:
                
                print("ZMQ: heartbeat haven't got reply, retrying...")
                
                self.last_heartbeat += 1.
                
                if (time() - self.last_reply) > 10.:
                    # reconnecting the socket as per
                    # the "lazy pirate" pattern (see the ZeroMQ guide)
                    print("ZMQ: connection lost, trying to reconnect")
                    self.poller.unregister(self.socket)
                    self.socket.close()
                    self.socket = None

                    self.socket_waits_reply = False
                    self.last_reply = time()
            else:
                self.send_heartbeat()
        
    def run(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://192.168.80.114:5556")
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        self.heartbeat_socket = self.context.socket(zmq.REQ)
        self.heartbeat_socket.connect("tcp://192.168.80.114:5557")
        self.last_heartbeat = float("-inf")
        self.socket_waits_reply = False
        
        self.poller = zmq.Poller()
        self.poller.register(self.socket)
        self.poller.register(self.heartbeat_socket)
        
        # self.compute_activations()
        self.receive_data()
        
    def receive(self, data_types={"data", "spike"}):
        try:
            message = self.socket.recv_multipart(zmq.NOBLOCK)
        except zmq.error.Again:
            print("ZMQ: ERROR")
            return None, None
        
        header = json.loads(message[1].decode('utf-8'))
        if "data" in data_types and header["type"] == "data":
            body = np.frombuffer(message[2], dtype=np.float32)
            return header, body
        elif "spike" in data_types and header["type"] == "spike":
            body = np.frombuffer(message[2], dtype=np.float32)
            return header, body
        elif "other" in data_types and header["type"] not in {"spike", "data"}:
            return header, body
        return None, None

    def receive_data(self):
        i = 0
        time_prev = time()
        time_start = time()
        while self.run_loop.value == 1:
            self.check_heartbeat()
            socks = dict(self.poller.poll(1))

            if self.socket in socks:
                header, body = self.receive({"spike"})
                if header is not None and header["type"] == "spike":
                    channel = int(header["spike"]["electrode"].split(" ")[-1]) - 1
                    # ^This depends on the electrode names in OpenEphys. Assumes interval [0, 15].
                    idx = channel * self.spike_history_len + ((self.spike_offsets[channel] + 1) % self.spike_history_len)
                    self.raw_spikes_buffer[idx] = time()
                    self.spike_offsets[channel] = self.spike_offsets[channel] + 1
                    self.spike_offsets[channel] = self.spike_offsets[channel] % self.spike_history_len
                    
                if (time() - time_prev) > self.dt:
                    time_prev = time()
                    i += 1
                    freqs = self._get_spike_frequencies()
                    
                    # print(i, time(), np.all(freqs < 10))
                    for channel in range(self.channels):
                        idx = channel * self.buffer_size + ((self.spike_frequency_offset.value + 1) % self.buffer_size)
                        self.spike_frequency_buffer[idx] = freqs[channel]
                    self.spike_frequency_offset.value += 1
                    self.spike_frequency_offset.value %= self.buffer_size
        
                if self.enough_time.value == 0 and (time() - time_start) > (self.dt * self.buffer_size) + 1:
                    self.enough_time.value = 1
                    
            elif self.heartbeat_socket in socks and self.socket_waits_reply:
                message = self.heartbeat_socket.recv()
                # print(f'ZMQ: Heartbeat reply: {message.decode("utf-8")}')
                if self.socket_waits_reply:
                    self.socket_waits_reply = False
                else:
                    print("ZMQ: Received reply before sending a message?")
            
        self.poller.unregister(self.socket)
        self.socket.close()
        self.poller.unregister(self.heartbeat_socket)
        self.heartbeat_socket.close()
                
    def stop(self):
        print("Stopping neuron stream")
        self.run_loop.value = 0
        sleep(1)
    
    def get_spike_frequencies_array(self, most_current=True):
        while self.enough_time.value == 0:
            sleep(0.1)
            print("Waiting to gather data.", end="\r")
            
        if self.enough_time.value == 1:
            print(" " * 30, end="\r")
        
        if most_current or self.last_spike_readout is None:
            self.last_spike_readout = np.zeros((self.channels, self.buffer_size))
            for channel in range(self.channels):
                for sample in range(self.buffer_size):
                    idx = channel * self.buffer_size + ((self.spike_frequency_offset.value + sample) % self.buffer_size)
                    self.last_spike_readout[channel, sample] = self.spike_frequency_buffer[idx]
                    
        return self.last_spike_readout

    def _get_spike_frequencies(self, channels=None):
        if channels is None:
            channels = np.arange(self.channels)
        frequencies = np.zeros(self.channels)
        
        for channel in channels:
            channel_events = []
            for event_idx in range(self.spike_history_len):
                idx = channel * self.spike_history_len + ((self.spike_offsets[channel] + event_idx) % self.spike_history_len)
                channel_events.append(self.raw_spikes_buffer[idx])
            channel_events = np.array(channel_events)
            channel_events = channel_events[channel_events != 0]
            if len(channel_events) <= 1:
                frequencies[channel] = 0
            else:
                time_delta = channel_events.max() - channel_events.min()
                frequencies[channel] = len(channel_events) / time_delta
        
        return frequencies


if __name__ == "__main__":
    try:
        neurons = NeuronStream(channels=32)
        neurons.start()
        
        sleep(1)
        frequencies = []
        timestamps = []
        time_start = time()
        while time() - time_start < 10:
            # n = neurons.get_raw_values_array()
            # print(n, n.shape)
            # frequencies.append(neurons.get_spike_frequencies()[:16])
            # timestamps.append(time())
            sleep(0.05)
            # print()
            # print(n[0, :])
            # raise KeyboardInterrupt
        
        from matplotlib import pyplot as plt
        # frequencies = np.stack(frequencies, axis=-1)
        frequencies = neurons.get_spike_frequencies_array()[:16]
        # frequencies = frequencies.clip(0, 100)
        # timestamps = np.array(timestamps) - min(timestamps)
        timestamps = np.linspace(0, 16/240*neurons.buffer_size, neurons.buffer_size)
        # timestamps = np.stack([timestamps] * 16, axis=0)
        print(frequencies.shape, timestamps.shape)
        fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(16, 16))
        axes = np.reshape(axes, -1)
        for i in range(len(frequencies)):
            axes[i].plot(timestamps, frequencies[i], label=str(i))
            axes[i].set_title("channel=" + str(i))
        # plt.legend()
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("frequency")
        
        plt.tight_layout()
        plt.savefig("freqs.png")
        
    except KeyboardInterrupt:
        print()
        neurons.stop()
            
    except Exception as e:
        print(e)