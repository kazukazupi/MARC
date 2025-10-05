from alg.coea.evolve_single import evolve_single
from utils import get_args, set_seed

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    evolve_single(args)
