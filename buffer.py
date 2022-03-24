import json
import copy
import json
import threading
import numpy as np
from collections import deque, namedtuple


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16,
                            np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ReplayBuffer(object):
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.lock = threading.Lock()
        if self.capacity is not None:
            self.observations = deque(maxlen=self.capacity)
            self.actions = deque(maxlen=self.capacity)
            self.rewrads = deque(maxlen=self.capacity)
            self.dones = deque(maxlen=self.capacity)
            self.behavior_policies = deque(maxlen=self.capacity)
        else:
            self.observations = deque()
            self.actions = deque()
            self.rewards = deque()
            self.dones = deque()
            self.behavior_policies = deque()

    def store(self, obs, act, rew, don, pol):
        self.lock.acquire()
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.dones.append(don)
        self.behavior_policies.append(pol)
        self.lock.release()

    def get_data(self, batch_size=None):
        self.lock.acquire()
        if batch_size is not None:
            observations = [self.observations.popleft() for _ in range(batch_size)]
            actions = [self.actions.popleft() for _ in range(batch_size)]
            rewards = [self.rewards.popleft() for _ in range(batch_size)]
            dones = [self.dones.popleft() for _ in range(batch_size)]
            behavior_policies = [self.behavior_policies.popleft() for _ in range(batch_size)]
        else:
            observations = copy.deepcopy(list(self.observations))
            actions = copy.deepcopy(list(self.actions))
            rewards = copy.deepcopy(list(self.rewards))
            dones = copy.deepcopy(list(self.dones))
            behavior_policies = copy.deepcopy(list(self.behavior_policies))
            self.clear()

        traj_data = namedtuple('traj_data', ['observations', 'actions', 'rewards', 'dones', 'behavior_polcies'])(
            observations, actions, rewards, dones, behavior_policies
        )
        self.lock.release()
        return traj_data

    def get_json_data(self, batch_size=None):
        traj_data = self.get_data(batch_size)
        traj_data = traj_data._asdict()
        json_data = json.dumps(traj_data, cls=NumpyEncoder)
        return json_data

    def clear(self):
        self.observations.clear()
        self.rewards.clear()
        self.actions.clear()
        self.dones.clear()
        self.behavior_policies.clear()

    def __len__(self):
        return len(self.dones)