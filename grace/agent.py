import numpy as np
from .mylog import Logger
log = Logger(__name__, 30).logger


class Agent:
    """Create an agent
    pose2d = [x, y, alpha]
    """
    def __init__(self, name, *args):
        log.debug(f"Init create Agent {name}")
        self.name = name
        if len(args) == 2:
            self.position = np.array(args[0])
            self.orientation = np.array(args[1])
        elif len(args) == 1:
            self.position, self.orientation = np.array(args[0], dtype=object)
        else:
            log.warning("Wrong number of arguments")
    
        self.params = {}
        self.WtE = {}
        assert len(self.position) == 3, "wrong size of agent pose"
        self.angle = None
        self._pose = None
        log.debug(f"Finish create Agent {name}")

    @property
    def mag(self):
        log.debug(f"getter of magnitude of {self.name}")
        return np.linalg.norm(self.position)

    @property
    def position(self):
        """I'm the 'position' property."""
        log.debug(f"getter of position of {self.name} called")
        return self._position

    @position.setter
    def position(self, value):
        log.debug(f"setter of position of {self.name} to {value}")
        assert len(value) == 3, "wrong size of value"
        self._position = np.array(value)
        # self.angle = value[2]

    @position.deleter
    def position(self):
        log.debug(f"deleter of position of {self.name} called")
        del self._position

    @property
    def angle(self):
        """I'm the 'angle' property."""
        log.debug(f"getter of angle of {self.name} called")
        return self._angle

    @angle.setter
    def angle(self, value):
        # assert len(value) == 1, "wrong size of value"
        log.debug(f"setter of angle of {self.name} to {value}")
        self._angle = check_is_radians(value)

    @angle.deleter
    def angle(self):
        log.debug(f"deleter of angle of {self.name} called")
        del self._angle

    def set_feature_params(self, params: dict):
        # {'proxemics':'p','gaze':[sign, angle]}
        self.params = params
        return self.willing_to_engage()

    def willing_to_engage(self):
        # return list where each element is a defined gaussian (per feature)
        log.debug(f"")
        log.debug(f"Agent: {self.name}")
        # log.debug(f'Gaze params {self.params["gaze"]}')
        for f in self.features_:
            log.debug(f"Set {f.name} to {self.params[f.name]}")
            self.WtE[f.name] = f.value(self.params[f.name])

def check_is_radians(angle):
    pass
    # if angle >= -2 * np.pi and angle <= 2 * np.pi:
        # log.debug(f"angle is in radians")
    # else:
        # raise ValueError(f"angle must be provided in radians!")
    # return angle

