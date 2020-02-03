import fwg
import numpy as np

if __name__ == "__main__":

    embeddings = [
        [(2.0, 4.0), (4.0, 8.0)],
        [(1.0, 2.0), (8.0, 20.0), (34.0, 90.0)],
        [(2.0, 4.0), (4.0, 8.0)]
    ]

    print(embeddings)

    distances = fwg.fwd(embeddings, embeddings, 50)

    print(np.matrix(distances))
