import quaternion as qt
import numpy as np
from .mylog import Logger
log = Logger(__name__, 30).logger


class Agent:
    """Create an agent
    orientation using quaternion library provide input as [x,y,z,w]
    """

    def __init__(self, name, *args):
        log.debug(f"Init create Agent {name}")
        self.name = name
        if len(args) == 2:
            self.position = np.array(args[0])
            orientation = np.array(args[1])
        elif len(args) == 1:
            self.position, orientation = np.array(args[0], dtype=object)
        else:
            raise ValueError("Wrong number of arguments")
            
        assert len(self.position) == 3, "wrong size of agent pose"
        assert len(orientation) == 4, "wrong size of agent orientation"

        self.orientation = qt.quaternion()
        self.orientation.x = orientation[0]
        self.orientation.y = orientation[1]
        self.orientation.z = orientation[2]
        self.orientation.w = orientation[3]
        
        norm = np.linalg.norm([self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w])
        assert np.isclose(norm, 1.0), f"Quaternion norm is {norm} and not 1.0"
        
        self.rpy = qt.as_euler_angles(self.orientation)
        self.params = {}
        self.WtE = {}
        self.angle = None
        self._pose = None
        log.debug(f"Finish create Agent {name}")

    def __str__(self):
        return f'Agent: {self.name} - {self.position} -  [x:{self.orientation.x} y:{self.orientation.y} z:{self.orientation.z} w:{self.orientation.w}] - rpy: {self.rpy}'

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
