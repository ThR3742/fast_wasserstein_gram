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

    for _ in range(10):
        distances = fwg.fwd(embeddings*5, embeddings*10, 50)
        print(len(distances))
        print(np.shape(np.matrix(distances)))
