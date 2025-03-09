import random
import shutil

import numpy as np

from alg.coea.structure import Structure


def test_score():

    save_path = "tests/temporary"

    try:
        n = 100
        id_list = list(range(n))
        scores = [random.random() for _ in range(n)]

        structure = Structure(save_path, np.zeros((5, 5)), np.zeros((5, 5)))

        for id_ in id_list:
            assert not structure.has_fought(id_)
        assert structure.fitness is None

        for id_, score in zip(id_list, scores):
            structure.set_score(id_, score)

        for id_ in id_list:
            assert structure.has_fought(id_)
        assert structure.fitness == np.mean(scores)

        d = 33
        for id_ in id_list[:d]:
            structure.delete_score(id_)

        for id_ in id_list:
            if id_ < d:
                assert not structure.has_fought(id_)
            else:
                assert structure.has_fought(id_)
        assert structure.fitness == np.mean(scores[d:])

        del structure
        structure_reloaded = Structure.from_save_path(save_path)

        for id_ in id_list:
            if id_ < d:
                assert not structure_reloaded.has_fought(id_)
            else:
                assert structure_reloaded.has_fought(id_)
        assert structure_reloaded.fitness == np.mean(scores[d:])

    finally:
        shutil.rmtree(save_path)
