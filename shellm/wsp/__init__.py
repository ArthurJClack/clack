"""weight-space programming: models as values you compute with."""
from .edit import Fact, edit_many, fact_logprob, key_covariance, rome
from .merge import dare, dare_ties, interference, linear, slerp, task_vector, ties, trim
from .recipe import loss, run_recipe, table
from .steer import ablate_direction, bake_steering, mean_residual, steering_vector
from .surgery import fake_quant, low_rank, prune, splice, stack_layers, transplant
from .tensors import Weights, load
