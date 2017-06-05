from newspaper import Article
import newspaper
from nltk import word_tokenize, sent_tokenize, Text
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
from nltk.sentiment import SentimentIntensityAnalyzer
import os



def getArticleUrls(site):
    # turned off momoize because
    # for some reason you cannot cache
    fullSiteBuild = newspaper.build(site, memoize_articles=False)
    return fullSiteBuild.articles


def processArticles(articles, requestedNumberOfWords):
    # download articles and extract sentences and words
    sentences = []
    words = []
    counter = 0
    countProcessedArticles = 0
    while len(words) < requestedNumberOfWords:
        counter += 1
        if articles == []:
            print('no more articles left')
            break
        if counter > 150:
            print("too many sites visited")
            break
        try:
            articles[0].download()
            articles[0].parse()
            articles[0].nlp()
            sentences += sent_tokenize(articles[0].text)
            words += word_tokenize(articles[0].text)
            countProcessedArticles += 1
            print('processed article: ' + articles[0].url)
        except:
            print('couldnt process article: ' + articles[0].url)
        articles.remove(articles[0])
    return [words, sentences, countProcessedArticles]


def mainInformationGatherer(siteNames, sitesInfo,
                            requestedNumberOfWords=10000):
    # schedule articles download and processing
    for siteName in siteNames:
        foundSite = False
        # check if info about this site already exists
        for siteInfo in sitesInfo:
            if siteInfo['name'] == siteName:
                foundSite = True
                siteToEdit = siteInfo
                break
        if not foundSite:
            siteToEdit = {}
            siteToEdit['name'] = siteName
            siteToEdit['articles'] = getArticleUrls(siteName)
            siteToEdit['words'] = []
            siteToEdit['sentences'] = []
            siteToEdit['processed articles'] = 0
            sitesInfo.append(siteToEdit)

        newWords, newSentences, newProcessedArticles =\
            processArticles(siteToEdit['articles'], requestedNumberOfWords)

        siteToEdit['words'] += newWords
        siteToEdit['sentences'] += newSentences
        siteToEdit['processed articles'] += newProcessedArticles
    return sitesInfo


def interpretInfo(siteInfo, plotAttitude=False):
    # rate how emotional are the sentences
    sia = SentimentIntensityAnalyzer()
    for site in siteInfo:
        compounds = []
        for sentence in site['sentences']:
            compounds.append(sia.polarity_scores(sentence)['compound'])
        compounds = np.array(compounds)
        site['emotional charge'] = np.mean(np.abs(compounds) ** 4)
        site['attitude'] = np.mean(compounds ** 3)
        if plotAttitude:
            plt.figure()
            plt.plot(walkingSum(compounds, 5))
            plt.title('sentence -> attitude')
            plt.draw()


def printSiteInfo(sitesInfo):
    plt.figure()
    plt.scatter([site['emotional charge'] for site in sitesInfo],
                [site['attitude'] for site in sitesInfo])
    plt.title('x: emotions, y:attitude')
    for site in sitesInfo:
        print(site['name'])
        print('articles:', site['processed articles'],
              '\twords:', len(site['words']),
              '\tsentences:', len(site['sentences']),
              '\temotions:', round(site['emotional charge'], 4),
              '\tattitude:', round(site['attitude'], 4))
        plt.annotate(site['name'],
                     (site['emotional charge'],
                      site['attitude']))
    plt.draw()


def diversity(words):
    # 0..1 the less the better diversity
    vocabulary = Text(words).vocab()
    sum = 0
    for w in vocabulary.items():
        sum += w[1] ** 2
    return sum / len(words) ** 2


def walkingSum(array, bufferLength):
    # makes the graph smoother
    return [sum(array[i:i + bufferLength])
            for i in range(len(array) - bufferLength + 1)]


def vocabulariesDistance(words1, words2, bigrams=False):
    # measures how different are two sets of words
    # if told to use bigrams substitute bigrams for words array
    if bigrams:
        words1 = [words1[i] + ' ' + words1[i + 1]
                  for i in range(len(words1) - 1)]
        words2 = [words2[i] + ' ' + words2[i + 1]
                  for i in range(len(words2) - 1)]

    allDifferentWords = set(words1).union(words2)
    # vocabulary is a dictionary: word -> number of occurences
    vocabulary1 = Text(words1).vocab()
    vocabulary2 = Text(words2).vocab()

    len1 = len(words1)
    len2 = len(words2)
    distance = 0
    for word in allDifferentWords:
        distance += np.abs(vocabulary1[word] / len1 - vocabulary2[word] / len2)
    return distance


def printClosestMatch(words, sitesInfo, bigrams=False):
    wordLengths = [len(siteInfo['words']) for siteInfo in sitesInfo]
    minLength = min(wordLengths)
    distances = []
    for siteInfo in sitesInfo:
        distances.append([vocabulariesDistance(words,
                                               siteInfo['words'][:minLength],
                                               bigrams=bigrams),
                          siteInfo['name']])
    for pair in sorted(distances):
        print(round(pair[0], 4), "\t", pair[1])


def compareArticleToOtherSites(url, sitesInfo, bigrams=False):
    if sitesInfo == 'DEFAULT':
        sitesInfo = load()
    article = Article(url)
    w, s, pa = processArticles([article], 1)
    articleInfo = {}
    articleInfo['name'] = url
    articleInfo['words'] = w
    articleInfo['sentences'] = s
    articleInfo['processed articles'] = pa

    baseUrl = url.rsplit('/')[2]
    urlClasses = {}
    path = os.path.dirname(__file__)
    path = os.path.join(path, 'sourcesUncut.csv')
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            urlClasses[row[0]] = row[1:]
    try:
        classification = ", ".join(urlClasses[baseUrl])
        print('\nclassification according to OpenSources.co:', classification)
    except:
        print('\nsite isnt listed in OpenSources.co      (thats good!)')

    if (article.authors == []):
        print('\nno authors found!')
    else:
        print('\nauthors:', article.authors)
    print('\nkeywords:', article.keywords)
    print()

    interpretInfo([articleInfo], plotAttitude=True)
    printSiteInfo(sitesInfo + [articleInfo])
    print("\nused words are most simiral to:   (in descending order)")
    printClosestMatch(articleInfo['words'], sitesInfo, bigrams=bigrams)


def save(sitesInfo):
    # saves sitesInfo to file
    with open('sitesInfo.pickle', 'wb') as handle:
        pickle.dump(sitesInfo, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load():
    # loads sitesInfo from file
    path = os.path.dirname(__file__)
    path = os.path.join(path, 'sitesInfo.pickle')
    with open(path, 'rb') as handle:
        sitesInfo = pickle.load(handle)
    return sitesInfo


siteNames = ['http://www.conservativeinfidel.com/',
             'http://thefreepatriot.org/',
             'http://ushealthyadvisor.com/',
             'http://www.naturalnews.com/',
             'https://www.infowars.com',
             'https://www.buzzfeed.com/',
             'http://edition.cnn.com/',
             'http://www.bbc.com/news/',
             'https://www.theguardian.com/international',
             'https://en.wikipedia.org/wiki']
