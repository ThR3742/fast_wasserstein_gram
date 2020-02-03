import fwg
import numpy as np

if __name__ == "__main__":

    embeddings_in = [
        [(2, 3), (4, 2), (5, 6)],
        [(2, 3), (4, 2), (5, 6), (8, 12)]
    ]
    embeddings_out = [
        [(3, 4), (5, 6)],
        [(3, 4), (5, 6)],
        [(3, 4)]
    ]

    print(embeddings_in)
    print(embeddings_out)

    gram = fwg.fwg(embeddings_in, embeddings_out, 37, 0.02)

    print(np.matrix(gram))
