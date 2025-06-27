import json
from typing import Union

class MambaConfig:
    """
    Configuration class for the Mamba model.

    This class centralizes all hyperparameters for the Mamba architecture,
    making it easy to create, manage, and share model configurations.
    """
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        pad_vocab_size_multiple: int = 8,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        use_fast_path: bool = True,
        **kwargs,  # To allow for future flexibility and backward compatibility
    ):
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.conv_bias = conv_bias
        self.bias = bias
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.use_fast_path = use_fast_path

        # Derived attributes
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == "auto":
            self.dt_rank = (self.d_model + 15) // 16 # math.ceil(self.d_model / 16)
        
        # --- FIX: Calculate and store the padded vocab size ---
        self.padded_vocab_size = self.vocab_size
        if self.padded_vocab_size % self.pad_vocab_size_multiple != 0:
            self.padded_vocab_size += self.pad_vocab_size_multiple - (self.padded_vocab_size % self.pad_vocab_size_multiple)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json_file(cls, json_file: str):
        """Creates an instance from a JSON file."""
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)