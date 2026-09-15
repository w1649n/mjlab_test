import numpy as np


class StandingGait:
    """A time-invariant four-foot contact schedule for quiet standing."""

    def __init__(self, nSegment: int, name: str = "Standing"):
        self.__nIterations = nSegment
        self.__name = name
        self.__mpc_table = [1 for _ in range(nSegment * 4)]

    def setIterations(self, iterationsPerMPC: int, currentIteration: int):
        del iterationsPerMPC, currentIteration

    def getContactState(self):
        return np.ones((4, 1), dtype=np.float32)

    def getCurrentContactState(self):
        return np.ones(4, dtype=np.float32)

    def getSwingState(self):
        return np.zeros((4, 1), dtype=np.float32)

    def getMpcTable(self):
        return self.__mpc_table

    def getCurrentGaitPhase(self):
        return 0

    def getCurrentSwingTime(self, dtMPC: float, leg: int):
        del dtMPC, leg
        return 0.0

    def getCurrentStanceTime(self, dtMPC: float, leg: int):
        del leg
        return dtMPC * self.__nIterations


class StoppingGait:
    """Finish only the active swing legs, without starting another pair."""

    def __init__(self, nSegment: int, swingSegments: int = 5):
        self.__nIterations = nSegment
        self.__swingSegments = swingSegments
        self.__contact = np.ones((4, 1), dtype=np.float32)
        self.__swing = np.zeros((4, 1), dtype=np.float32)
        self.__mpc_table = [1 for _ in range(nSegment * 4)]

    def configure(self, activeSwing, swingProgress, footContacts):
        active = np.asarray(activeSwing, dtype=bool).reshape(4)
        progress = np.asarray(swingProgress, dtype=np.float32).reshape(4)
        contacts = np.asarray(footContacts, dtype=bool).reshape(4)
        self.__contact.fill(1)
        self.__swing.fill(0)
        for leg in range(4):
            if active[leg] and not contacts[leg]:
                self.__contact[leg, 0] = 0.0
                self.__swing[leg, 0] = max(float(progress[leg]), 1e-6)

        remaining_segments = np.zeros(4, dtype=int)
        for leg in range(4):
            if active[leg] and not contacts[leg]:
                # The current row remains non-contact until the debounced
                # sensor confirms touchdown.  Keep one future segment of
                # landing intent during terrain search: removing that intent
                # from the complete horizon leaves only a diagonal support
                # pair and destabilizes an otherwise recoverable stop.
                remaining_segments[leg] = max(
                    1,
                    int(np.ceil((1.0 - min(float(progress[leg]), 1.0))
                                * self.__swingSegments)),
                )
        for segment in range(self.__nIterations):
            for leg in range(4):
                self.__mpc_table[segment * 4 + leg] = int(
                    segment >= remaining_segments[leg]
                )

    def setIterations(self, iterationsPerMPC: int, currentIteration: int):
        del iterationsPerMPC, currentIteration

    def getContactState(self):
        return self.__contact

    def getCurrentContactState(self):
        return self.__contact.reshape(4).copy()

    def getSwingState(self):
        return self.__swing

    def getMpcTable(self):
        return self.__mpc_table

    def getCurrentGaitPhase(self):
        return 0

    def getCurrentSwingTime(self, dtMPC: float, leg: int):
        del leg
        return dtMPC * self.__swingSegments

    def getCurrentStanceTime(self, dtMPC: float, leg: int):
        del leg
        # The zero-velocity stop still needs the longer capture-step correction
        # while the body sheds its residual momentum. Closed-loop gates showed
        # that shortening this to the source trot stance time destabilizes the
        # final diagonal support phase.
        return dtMPC * self.__nIterations

class OffsetDurationGait:
    """
    trotting, bounding, pronking
    jumping, galloping, standing
    trotRunning, walking, walking2
    pacing
    """
    def __init__(self, nSegment:int, offset:np.ndarray, durations:np.ndarray, name:str, cycle_segments=None):

        # The prediction horizon and full gait cycle need not have equal lengths.
        cycle_segments = nSegment if cycle_segments is None else float(cycle_segments)
        if not np.isfinite(cycle_segments) or cycle_segments <= 0:
            raise ValueError("cycle_segments must be finite and positive")
        self.__horizon = nSegment

        # offset in mpc segments
        self.__offsets = offset.flatten() * (cycle_segments / nSegment)
        # duration of step in mpc segments
        self.__durations = durations.flatten() * (cycle_segments / nSegment)
        # offsets in phase (0 to 1)
        self.__offsetsFloat = offset / nSegment
        # durations in phase (0 to 1)
        self.__durationsFloat = durations / nSegment
        self.__nIterations = cycle_segments
        self.__name = name
        self.__stance = self.__durations[0]
        self.__swing = cycle_segments - self.__stance
        self.__mpc_table = [0 for _ in range(nSegment*4)]

    def setIterations(self, iterationsPerMPC:int, currentIteration:int):
        # Match the original C++ integer division: the MPC horizon advances by
        # one segment only after ``iterationsPerMPC`` controller ticks.
        self.__iteration = (currentIteration // iterationsPerMPC) % self.__nIterations
        self.__phase = float(currentIteration % (iterationsPerMPC * self.__nIterations)) / float(iterationsPerMPC * self.__nIterations)

    def getContactState(self):
        progress = self.__phase - self.__offsetsFloat

        for i in range(4):
            if progress[i] < 0:
             progress[i] += 1.0

            if progress[i] > self.__durationsFloat[i]:
                progress[i] = 0.0
            else:
                progress[i] = progress[i] / self.__durationsFloat[i]
            
        # print("contact state: %.3f %.3f %.3f %.3f"%(progress[0], progress[1], progress[2], progress[3]))
        return progress[:, None] # convert to matrix

    def getSwingState(self):
        swing_offset = self.__offsetsFloat + self.__durationsFloat
        for i in range(4):
            if swing_offset[i] > 1:
                swing_offset[i] -= 1.0
        swing_duration = np.ones_like(self.__durationsFloat) - self.__durationsFloat

        progress = self.__phase - swing_offset

        for i in range(4):
            if progress[i] < 0:
                progress[i] += 1.0

            if progress[i] > swing_duration[i]:
                progress[i] = 0.0
            else:
                if swing_duration[i] == 0.0:
                    progress[i] = 0.0
                else:
                    progress[i] = progress[i] / swing_duration[i]

        # print("swing state: %.3f %.3f %.3f %.3f"%(progress[0], progress[1], progress[2], progress[3]))
        return progress[:,None]

    def getMpcTable(self):
        # print("MPC table:")
        for i in range(self.__horizon):
    
            iter = (i + self.__iteration + 1) % self.__nIterations
            progress = iter - self.__offsets
            for j in range(4):
                if progress[j] < 0:
                    progress[j] += self.__nIterations
                if progress[j] < self.__durations[j]:
                    self.__mpc_table[i * 4 + j] = 1
                else:
                    self.__mpc_table[i * 4 + j] = 0
            # print("%d "% self.__mpc_table[i*4 + j], end="")
        
        return self.__mpc_table

    def getCurrentGaitPhase(self):
        return self.__iteration

    def getCurrentContactState(self):
        progress = (self.__iteration - self.__offsets) % self.__nIterations
        return (progress < self.__durations).astype(np.float32)

    def getCurrentSwingTime(self, dtMPC:float, leg:int):
        return dtMPC * self.__swing

    def getCurrentStanceTime(self, dtMPC:float, leg:int):
        return dtMPC * self.__stance
