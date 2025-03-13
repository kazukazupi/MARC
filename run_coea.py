from alg.coea import evolve
from utils import get_args, set_seed

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    evolve(args)
