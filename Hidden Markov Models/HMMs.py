__author__ = 'Schmidtz'

class GeneralHMMs():
    def __init__(self,start_prob=None, transition_mtx=None, obsevation_prob=None,algorithm="viterbi"):
        self.start_prob = start_prob
        self.transision_mtx = transition_mtx
        self.observation_prob = obsevation_prob
        self.algorithm = algorithm
