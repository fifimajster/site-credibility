import dodgyness
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('\n\n\n')
    sitesInfo = dodgyness.load()
    url = sys.argv[1]
    if len(sys.argv) == 3 and sys.argv[2] == 'bigrams':
        bigramOpt = True
    else:
        bigramOpt = False
    dodgyness.compareArticleToOtherSites(url, sitesInfo, bigrams=bigramOpt)
    plt.show()
