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

    expected = np.matrix([[ 0., 32.12016921, 0.],
                         [32.12016921, 0., 32.12016921],
                         [0., 32.12016921, 0.]])

    assert np.isclose(np.linalg.norm(np.matrix(distances)-expected), 0, 1e-7)


    for _ in range(10):
        distances = fwg.fwd(embeddings*3, embeddings*2, 50)
    print(distances)
    print(np.shape(distances))
