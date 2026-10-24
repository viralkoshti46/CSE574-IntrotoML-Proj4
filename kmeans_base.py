__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

from common import RESULTS_DIR, dataset, kmeans, read_images, write_centers

NUMBER_OF_STARTS = 5
MAX_ITERATIONS = 35
BATCH_SIZE = 1000
RANDOM_SEED = 42


def run(k):
    images = read_images(dataset("train-images.idx3-ubyte"))
    distortion, centers, winning_start = kmeans(
        images, k, NUMBER_OF_STARTS, MAX_ITERATIONS, BATCH_SIZE, RANDOM_SEED
    )
    print(f"Best starting run: {winning_start}; distortion: {distortion:.3f}")
    print("Best cluster centers (each row is one center):")
    print(centers)
    write_centers(k, centers)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"kmeans_{k}_summary.txt").write_text(
        f"Clusters: {k}\nRandom starts: {NUMBER_OF_STARTS}\nBest run: {winning_start}\nTotal distortion: {distortion:.8f}\n"
    )
    print(f"Centers saved to results/centers_{k}.xlsx")
