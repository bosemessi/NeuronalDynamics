import numpy as np
from scipy import linalg
import pickle
import gzip
from pkg_resources import resource_filename
import sys
import matplotlib
import matplotlib.pyplot as plt

class HopfieldNetwork:
    """Implements a Hopfield network.

    Attributes:
        nrOfNeurons (int): Number of neurons
        weights (numpy.ndarray): nrOfNeurons x nrOfNeurons matrix of weights
        state (numpy.ndarray): current network state. matrix of shape (nrOfNeurons, nrOfNeurons)
    """

    def __init__(self, nr_neurons):
        """
        Constructor

        Args:
            nr_neurons (int): Number of neurons. Use a square number to get the
            visualizations properly
        """
        # math.sqrt(nr_neurons)
        self.nrOfNeurons = nr_neurons
        # initialize with random state
        self.state = 2 * np.random.randint(0, 2, self.nrOfNeurons) - 1
        # initialize random weights
        self.weights = 0
        self.reset_weights()
        self._update_method = _get_sign_update_function()

    def reset_weights(self):
        """
        Resets the weights to random values
        """
        self.weights = 1.0 / self.nrOfNeurons * \
            (2 * np.random.rand(self.nrOfNeurons, self.nrOfNeurons) - 1)


    def set_dynamics_sign_sync(self):
        """
        sets the update dynamics to the synchronous, deterministic g(h) = sign(h) function
        """
        self._update_method = _get_sign_update_function()


    def set_dynamics_sign_async(self):
        """
        Sets the update dynamics to the g(h) =  sign(h) functions. Neurons are updated asynchronously:
        In random order, all neurons are updated sequentially
        """
        self._update_method = _get_async_sign_update_function()


    def set_dynamics_to_user_function(self, update_function):
        """
        Sets the network dynamics to the given update function

        Args:
            update_function: upd(state_t0, weights) -> state_t1.
                Any function mapping a state s0 to the next state
                s1 using a function of s0 and weights.
        """
        self._update_method = update_function


    def store_patterns(self, pattern_list):
        """
        Learns the patterns by setting the network weights. The patterns
        themselves are not stored, only the weights are updated!
        self connections are set to 0.

        Args:
            pattern_list: a nonempty list of patterns.
        """
        all_same_size_as_net = all(len(p.flatten()) == self.nrOfNeurons for p in pattern_list)
        if not all_same_size_as_net:
            errMsg = "Not all patterns in pattern_list have exactly the same number of states " \
                     "as this network has neurons n = {0}.".format(self.nrOfNeurons)
            raise ValueError(errMsg)
        self.weights = np.zeros((self.nrOfNeurons, self.nrOfNeurons))
        # textbook formula to compute the weights:
        for p in pattern_list:
            p_flat = p.flatten()
            for i in range(self.nrOfNeurons):
                for k in range(self.nrOfNeurons):
                    self.weights[i, k] += p_flat[i] * p_flat[k]
        self.weights /= self.nrOfNeurons
        # no self connections:
        np.fill_diagonal(self.weights, 0)


    def set_state_from_pattern(self, pattern):
        """
        Sets the neuron states to the pattern pixel. The pattern is flattened.

        Args:
            pattern: pattern
        """
        self.state = pattern.copy().flatten()


    def iterate(self):
        """Executes one timestep of the dynamics"""
        self.state = self._update_method(self.state, self.weights)


    def run(self, nr_steps=5):
        """Runs the dynamics.

        Args:
            nr_steps (float, optional): Timesteps to simulate
        """
        for i in range(nr_steps):
            # run a step
            self.iterate()

    def run_with_monitoring(self, nr_steps=5):
        """
        Iterates at most nr_steps steps. records the network state after every
        iteration

        Args:
            nr_steps:

        Returns:
            a list of 2d network states
        """
        states = list()
        states.append(self.state.copy())
        for i in range(nr_steps):
            # run a step
            self.iterate()
            states.append(self.state.copy())
        return states

def _get_sign_update_function():
    """
    for internal use

    Returns:
        A function implementing a synchronous state update using sign(h)
    """
    def upd(state_s0, weights):
        h = np.sum(weights * state_s0, axis=1)
        s1 = np.sign(h)
        # by definition, neurons have state +/-1. If the
        # sign function returns 0, we set it to +1
        idx0 = s1 == 0
        s1[idx0] = 1
        return s1
    return upd

def _get_async_sign_update_function():
    def upd(state_s0, weights):
        random_neuron_idx_list = np.random.permutation(len(state_s0))
        state_s1 = state_s0.copy()
        for i in range(len(random_neuron_idx_list)):
            rand_neuron_i = random_neuron_idx_list[i]
            h_i = np.dot(weights[:, rand_neuron_i], state_s1)
            s_i = np.sign(h_i)
            if s_i == 0:
                s_i = 1
            state_s1[rand_neuron_i] = s_i
        return state_s1
    return upd

class PatternFactory:
    """
    Creates square patterns of size pattern_length x pattern_width
    If pattern length is omitted, square patterns are produced
    """
    def __init__(self, pattern_length, pattern_width=None):
        """
        Constructor
        Args:
            pattern_length: the length of a pattern
            pattern_width: width or None. If None, patterns are squares of size (pattern_length x pattern_length)
        """
        self.pattern_length = pattern_length
        self.pattern_width = pattern_length if pattern_width is None else pattern_width

    def create_random_pattern(self, on_probability=0.5):
        """
        Creates a pattern_length by pattern_width 2D random pattern
        Args:
            on_probability:

        Returns:
            a new random pattern
        """
        p = np.random.binomial(1, on_probability, self.pattern_length * self.pattern_width)
        p = p * 2 - 1  # map {0, 1} to {-1 +1}
        return p.reshape((self.pattern_length, self.pattern_width))


    def create_random_pattern_list(self, nr_patterns, on_probability=0.5):
        """
        Creates a list of nr_patterns random patterns
        Args:
            nr_patterns: length of the new list
            on_probability:

        Returns:
            a list of new random patterns of size (pattern_length x pattern_width)
        """
        p = list()
        for i in range(nr_patterns):
            p.append(self.create_random_pattern(on_probability))
        return p


    def create_row_patterns(self, nr_patterns=None):
        """
        creates a list of n patterns, the i-th pattern in the list
        has all states of the i-th row set to active.
        This is convenient to create a list of orthogonal patterns which
        are easy to visually identify

        Args:
            nr_patterns:

        Returns:
            list of orthogonal patterns
        """
        n = self.pattern_width if nr_patterns is None else nr_patterns
        pattern_list = []
        for i in range(n):
            p = self.create_all_off()
            p[i, :] = np.ones((1, self.pattern_length))
            pattern_list.append(p)
        return pattern_list


    def create_all_on(self):
        """
        Returns:
            2d pattern, all pixels on
        """
        return np.ones((self.pattern_length, self.pattern_width), int)

    def create_all_off(self):
        """
        Returns:
            2d pattern, all pixels off
        """
        return -1 * np.ones((self.pattern_length, self.pattern_width), int)


    def create_checkerboard(self):
        """
        creates a checkerboard pattern of size (pattern_length x pattern_width)
        Returns:
            checkerboard pattern
        """
        pw = np.ones(self.pattern_length, int)
        # set every second value to -1
        pw[1::2] = -1
        pl = np.ones(self.pattern_width, int)
        # set every second value to -1
        pl[1::2] = -1
        t = linalg.toeplitz(pw, pl)
        t = t.reshape((self.pattern_length, self.pattern_width))
        return t


    def create_L_pattern(self, l_width=1):
        """
        creates a pattern with column 0 (left) and row n (bottom) set to +1.
        Increase l_width to set more columns and rows (default is 1)

        Args:
            l_width (int): nr of rows and columns to set

        Returns:
            an L shaped pattern.
        """
        l_pat = -1 * np.ones((self.pattern_length, self.pattern_width), int)
        for i in range(l_width):
            l_pat[-i - 1, :] = np.ones(self.pattern_length, int)
            l_pat[:, i] = np.ones(self.pattern_length, int)
        return l_pat


    def reshape_patterns(self, pattern_list):
        """
        reshapes all patterns in pattern_list to have shape = (self.pattern_length, self.pattern_width)

        Args:
            self:
            pattern_list:

        Returns:

        """
        new_shape = (self.pattern_length, self.pattern_width)
        return reshape_patterns(pattern_list, new_shape)

def reshape_patterns(pattern_list, shape):
    """
    reshapes each pattern in pattern_list to the given shape

    Args:
        pattern_list:
        shape:

    Returns:

    """
    reshaped_patterns = [p.reshape(shape) for p in pattern_list]
    return reshaped_patterns

def get_pattern_diff(pattern1, pattern2, diff_code=0):
    """
    Creates a new pattern of same size as the two patterns.
    the diff pattern has the values pattern1 = pattern2 where the two patterns have
    the same value. Locations that differ between the two patterns are set to
    diff_code (default = 0)

    Args:
        pattern1:
        pattern2:
        diff_code: the values of the new pattern, at locations that differ between
        the two patterns are set to diff_code.
    Returns:
        the diff pattern.
    """
    if pattern1.shape != pattern2.shape:
        raise ValueError("patterns are not of equal shape")
    diffs = np.multiply(pattern1, pattern2)
    pattern_with_diffs = np.where(diffs < 0, diff_code, pattern1)
    return pattern_with_diffs

def flip_n(template, nr_of_flips):
    """
    makes a copy of the template pattern and flips
    exactly n randomly selected states.
    Args:
        template:
        nr_of_flips:
    Returns:
        a new pattern
    """
    n = np.prod(template.shape)
    # pick nrOfMutations indices (without replacement)
    idx_reassignment = np.random.choice(n, nr_of_flips, replace=False)
    linear_template = template.flatten()
    linear_template[idx_reassignment] = -linear_template[idx_reassignment]
    return linear_template.reshape(template.shape)

def get_noisy_copy(template, noise_level):
    """
    Creates a copy of the template pattern and reassigns N pixels. N is determined
    by the noise_level
    Note: reassigning a random value is not the same as flipping the state. This
    function reassigns a random value.

    Args:
        template:
        noise_level: a value in [0,1]. for 0, this returns a copy of the template.
        for 1, a random pattern of the same size as template is returned.
    Returns:

    """
    if noise_level == 0:
        return template.copy()
    if noise_level < 0 or noise_level > 1:
        raise ValueError("noise level is not in [0,1] but {}0".format(noise_level))
    linear_template = template.copy().flatten()
    n = np.prod(template.shape)
    nr_mutations = int(round(n * noise_level))
    idx_reassignment = np.random.choice(n, nr_mutations, replace=False)
    rand_values = np.random.binomial(1, 0.5, nr_mutations)
    rand_values = rand_values * 2 - 1  # map {0,1} to {-1, +1}
    linear_template[idx_reassignment] = rand_values
    return linear_template.reshape(template.shape)

def compute_overlap(pattern1, pattern2):
    """
    compute overlap

    Args:
        pattern1:
        pattern2:

    Returns: Overlap between pattern1 and pattern2

    """
    shape1 = pattern1.shape
    if shape1 != pattern2.shape:
        raise ValueError("patterns are not of equal shape")
    dot_prod = np.dot(pattern1.flatten(), pattern2.flatten())
    return float(dot_prod) / (np.prod(shape1))

def compute_overlap_list(reference_pattern, pattern_list):
    """
    Computes the overlap between the reference_pattern and each pattern
    in pattern_list

    Args:
        reference_pattern:
        pattern_list: list of patterns

    Returns:
        A list of the same length as pattern_list
    """
    overlap = np.zeros(len(pattern_list))
    for i in range(0, len(pattern_list)):
        overlap[i] = compute_overlap(reference_pattern, pattern_list[i])
    return overlap

def compute_overlap_matrix(pattern_list):
    """
    For each pattern, it computes the overlap to all other patterns.

    Args:
        pattern_list:

    Returns:
        the matrix m(i,k) = overlap(pattern_list[i], pattern_list[k]
    """
    nr_patterns = len(pattern_list)
    overlap = np.zeros((nr_patterns, nr_patterns))
    for i in range(nr_patterns):
        for k in range(i, nr_patterns):
            if i == k:
                overlap[i, i] = 1  # no need to compute the overlap with itself
            else:
                overlap[i, k] = compute_overlap(pattern_list[i], pattern_list[k])
                overlap[k, i] = overlap[i, k]  # because overlap is symmetric
    return overlap

def load_alphabet():
    """Load alphabet dict from the file
    ``data/alphabet.pickle.gz``, which is included in
    the neurodynex3 release.

    Returns:
        dict: Dictionary of 10x10 patterns

    Raises:
        ImportError: Raised if ``neurodynex``
            can not be imported. Please install
            `neurodynex <pypi.python.org/pypi/neurodynex/>`_.
    """
    # Todo: consider removing the zip file and explicitly store the strings here.
    file_str = "../data/alphabet.pickle.gz"

    try:
        file_name = "../data/alphabet.pickle.gz"  ### resource_filename("neurodynex3", file_str)
    except ImportError:
        raise ImportError(
            "Could not import data file %s. " % file_str +
            "Make sure the pypi package `neurodynex` is installed!"
        )

    with gzip.open("%s" % file_name) as f:
        if sys.version_info < (3, 0, 0):
            # python2 pickle.loads has no attribute "encoding"
            abc_dict = pickle.load(f)
        else:
            # latin1 is required for python3 compatibility
            abc_dict = pickle.load(f, encoding="latin1")

    # shape the patterns and provide upper case keys
    ABC_dict = dict()
    for key in abc_dict:
        ABC_dict[key.upper()] = abc_dict[key].reshape((10, 10))
    return ABC_dict

def plot_pattern(pattern, reference=None, color_map="brg", diff_code=0):
    """
    Plots the pattern. If a (optional) reference pattern is provided, the pattern is  plotted
     with differences highlighted

    Args:
        pattern (numpy.ndarray): N by N pattern to plot
        reference (numpy.ndarray):  optional. If set, differences between pattern and reference are highlighted
    """
    plt.figure()
    if reference is None:
        p = pattern
        overlap = 1
    else:
        p = get_pattern_diff(pattern, reference, diff_code)
        overlap = compute_overlap(pattern, reference)

    plt.imshow(p, interpolation="nearest", cmap=color_map)
    if reference is not None:
        plt.title("m = {:0.2f}".format(round(overlap, 2)))
    plt.axis("off")
    plt.show()

def plot_overlap_matrix(overlap_matrix, color_map="bwr"):
    """
    Visualizes the pattern overlap

    Args:
        overlap_matrix:
        color_map:

    """
    fig,ax = plt.subplots()
    im = ax.imshow(overlap_matrix, interpolation="nearest", cmap=color_map)
    ax.set_title("pattern overlap m(i,k)")
    ax.set_xlabel("pattern k")
    ax.set_ylabel("pattern i")
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax)
    plt.show()

def plot_pattern_list(pattern_list, color_map="brg"):
    """
    Plots the list of patterns

    Args:
        pattern_list:
        color_map:

    Returns:

    """
    f, ax = plt.subplots(1, len(pattern_list))
    if len(pattern_list) == 1:
        ax = [ax]  # for n=1, subplots() does not return a list
    _plot_list(ax, pattern_list, None, "P{0}", color_map)
    plt.show()

def _plot_list(axes_list, state_sequence, reference=None, title_pattern="S({0})", color_map="brg"):
    """
    For internal use.
    Plots all states S(t) or patterns P in state_sequence.
    If a (optional) reference pattern is provided, the patters are  plotted with differences highlighted

    Args:
        state_sequence: (list(numpy.ndarray))
        reference: (numpy.ndarray)
        title_pattern (str) pattern injecting index i
    """
    for i in range(len(state_sequence)):
        if reference is None:
            p = state_sequence[i]
        else:
            p = get_pattern_diff(state_sequence[i], reference, diff_code=-0.2)
        if np.max(p) == np.min(p):
            axes_list[i].imshow(p, interpolation="nearest", cmap='RdYlBu')
        else:
            axes_list[i].imshow(p, interpolation="nearest", cmap=color_map)
        axes_list[i].set_title(title_pattern.format(i))
        axes_list[i].axis("off")

def plot_state_sequence_and_overlap(state_sequence, pattern_list, reference_idx, color_map="brg", suptitle=None):
    """
    For each time point t ( = index of state_sequence), plots the sequence of states and the overlap (barplot)
    between state(t) and each pattern.

    Args:
        state_sequence: (list(numpy.ndarray))
        pattern_list: (list(numpy.ndarray))
        reference_idx: (int) identifies the pattern in pattern_list for which wrong pixels are colored.
    """
    if reference_idx is None:
        reference_idx = 0
    reference = pattern_list[reference_idx]
    f, ax = plt.subplots(2, len(state_sequence))
    if len(state_sequence) == 1:
        ax = [ax]
    print()
    _plot_list(ax[0, :], state_sequence, reference, "S{0}", color_map)
    for i in range(len(state_sequence)):
        overlap_list = compute_overlap_list(state_sequence[i], pattern_list)
        ax[1, i].bar(range(len(overlap_list)), overlap_list)
        ax[1, i].set_title("m = {1}".format(i, round(overlap_list[reference_idx], 2)))
        ax[1, i].set_ylim([-1, 1])
        ax[1, i].get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
        if i > 0:  # show lables only for the first subplot
            ax[1, i].set_xticklabels([])
            ax[1, i].set_yticklabels([])
    if suptitle is not None:
        f.suptitle(suptitle)
    plt.show()

def plot_network_weights(hopfield_network, color_map="jet"):
    """
    Visualizes the network's weight matrix

    Args:
        hopfield_network:
        color_map:

    """

    plt.figure()
    plt.imshow(hopfield_network.weights, interpolation="nearest", cmap=color_map)
    plt.colorbar()
