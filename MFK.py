def clustering(data):
    firstClusterOld = []
    secondClusterOld = []
    firstClusterNew = []
    secondClusterNew = []
    firstCenter = 5
    secondCenter = 15
    while True:
        for company in companyMeans:
            if abs(company[0] - firstCenter) > abs(company[0] - secondCenter):
                secondClusterNew.append(company)
            else:
                firstClusterNew.append(company)
        if firstClusterNew == firstClusterOld:
            break
        firstClusterOld = copy.deepcopy(firstClusterNew)
        secondClusterOld = copy.deepcopy(secondClusterNew)
        firstClusterNew.clear()
        secondClusterNew.clear()
        firstCenter = sum([company[0] for company in firstClusterOld]) / len([company[0] for company in firstClusterOld])
        secondCenter = sum([company[0] for company in secondClusterOld]) / len([company[0] for company in secondClusterOld])
    return (firstClusterOld, secondClusterOld)

import math
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
from fitter import Fitter

# Задание 1.
flights = pd.read_csv('D:\\flights_NY.csv')
print('Columns with NaNs:')
for item in flights:
    if flights[item].isnull().values.any():
        print(item)
fixedFlights = flights.dropna()
print(len(flights) - len(fixedFlights))
NaNRows = flights.drop(fixedFlights.index)

# Задание 2.
depDelaysDesc = fixedFlights['dep_delay'].describe()
arrDelaysDesc = fixedFlights['arr_delay'].describe()
depIntQuartileRange = 1.5 * (depDelaysDesc['75%'] - depDelaysDesc['25%'])
depBorders = (depDelaysDesc['25%'] - depIntQuartileRange, depDelaysDesc['75%'] + depIntQuartileRange)
arrIntQuartileRange = 1.5 * (arrDelaysDesc['75%'] - arrDelaysDesc['25%'])
arrBorders = (arrDelaysDesc['25%'] - arrIntQuartileRange, arrDelaysDesc['75%'] + arrIntQuartileRange)

bins = np.linspace(min(depBorders[0], arrBorders[0]), max(depBorders[1], arrBorders[1]), 100)
delays = fixedFlights[['dep_delay', 'arr_delay']]
plt.xlabel('Arrival/departure delay')
plt.ylabel('Number of delays')
delays.plot(kind = 'hist', alpha = 0.3, bins = bins, y = ['dep_delay', 'arr_delay'], xlabel = 'Frequency of arrival/departure delays, %', ylabel = 'Number of delays', density = True)
plt.savefig('D:\\MFK\\depArrHist.jpg')
plt.clf()
pd.set_option('display.max_rows', None)

lowAnomaliesArrBins = np.linspace(min(fixedFlights['arr_delay']), arrBorders[0], 50)
lowAnomDelaysArr = fixedFlights['arr_delay']
lowAnomDelaysArr.plot(kind = 'hist', bins = lowAnomaliesArrBins, color = 'orange', xlabel = 'Arrival delay', ylabel = 'Number of outliers')
plt.savefig('D:\\MFK\\lowAnomaliesArrHist.jpg')
plt.clf()
print('Low anomalies arrival counts:')
print(fixedFlights[fixedFlights['arr_delay'] < arrBorders[0]]['arr_delay'].value_counts().to_string())

highAnomaliesArrBins = np.linspace(arrBorders[1], max(fixedFlights['arr_delay']), 100)
highAnomDelaysArr = fixedFlights['arr_delay']
highAnomDelaysArr.plot(kind = 'hist', bins = highAnomaliesArrBins, color = 'orange', xlabel = 'Arrival delay', ylabel = 'Number of outliers')
plt.savefig('D:\\MFK\\highAnomaliesArrHist.jpg')
plt.clf()
highAnomArrFile = open('D:\\MFK\\highAnomArrFile.txt', 'w')
highAnomArrFile.write(fixedFlights[fixedFlights['arr_delay'] > arrBorders[1]]['arr_delay'].value_counts().to_string())
highAnomArrFile.close()

lowAnomaliesDepBins = np.linspace(min(fixedFlights['dep_delay']), depBorders[0], 30)
lowAnomDelaysDep = fixedFlights['dep_delay']
lowAnomDelaysDep.plot(kind = 'hist', bins = lowAnomaliesDepBins, color = 'Blue', xlabel = 'Departure delay', ylabel = 'Number of outliers')
plt.savefig('D:\\MFK\\lowAnomaliesDepHist.jpg')
plt.clf()
print('Low anomalies departure counts:')
print(fixedFlights[fixedFlights['dep_delay'] < depBorders[0]]['dep_delay'].value_counts())

highAnomaliesDepBins = np.linspace(arrBorders[1], max(fixedFlights['dep_delay']), 100)
highAnomDelaysDep = fixedFlights['dep_delay']
highAnomDelaysDep.plot(kind = 'hist', bins = highAnomaliesDepBins, color = 'Blue', xlabel = 'Departure delay', ylabel = 'Number of outliers')
plt.savefig('D:\\MFK\\highAnomaliesDepHist.jpg')
plt.clf()
highAnomDepFile = open('D:\\MFK\\highAnomDepFile.txt', 'w')
highAnomDepFile.write(fixedFlights[fixedFlights['dep_delay'] > depBorders[1]]['dep_delay'].value_counts().to_string())
highAnomDepFile.close()
pd.reset_option('all')

# Задание 3.
print('Departure delays description:')
print(depDelaysDesc['mean'])
print(depDelaysDesc['std'])
print(depDelaysDesc['50%'])
print('Arrival delays description:')
print(arrDelaysDesc['mean'])
print(arrDelaysDesc['std'])
print(arrDelaysDesc['50%'])

# Задание 4.
companyDelays = list()
descs = dict()
num = 0
colors = ['blue', 'red', 'yellow', 'red', 'orange', 'black', 'green', 'cian', 'brown', 'magenta', 'purple', 'pink' , 'gray', 'olive', 'plum', 'aquamarine']
companyMeans = tuple()
for company in fixedFlights['carrier'].unique():
    desc = fixedFlights[fixedFlights['carrier'] == company]['dep_delay'].describe()
    descs[company] = desc
    companyDelays.append((desc['mean'], company))
    border = 1.96 * desc['std'] / math.sqrt(desc['count'])
    plt.plot((desc['mean'] - border, desc['mean'] + border), (num, num), label = company)
    plt.legend(loc = 'best')
    num += 1
plt.savefig('D:\\MFK\\ints.jpg')
plt.clf()
companyMeans = [(descs[key]['mean'], key) for key in descs]
print(sorted(companyMeans))

# Задание 5.
AADelays = pd.Series(fixedFlights[fixedFlights['carrier'] == 'AA']['dep_delay'].values)
DLDelays = pd.Series(fixedFlights[fixedFlights['carrier'] == 'DL']['dep_delay'].values)
print(scp.stats.ttest_ind(AADelays, DLDelays, equal_var = False))

# Задание 6.
JFKDelays = pd.Series(fixedFlights[fixedFlights['origin'] == 'JFK']['dep_delay'].values)
LGADelays = pd.Series(fixedFlights[fixedFlights['origin'] == 'LGA']['dep_delay'].values)
EWRDelays = pd.Series(fixedFlights[fixedFlights['origin'] == 'EWR']['dep_delay'].values)
print(scp.stats.ttest_ind(JFKDelays, LGADelays))
print(scp.stats.ttest_ind(JFKDelays, EWRDelays))
print(scp.stats.ttest_ind(EWRDelays, LGADelays))

# Задание 7.
delays = fixedFlights[fixedFlights['dep_delay'] > 0]['dep_delay']
description = delays.describe()
mean = description['mean']
var = description['std'] ** 2
values = pd.Series(delays.values)
likelihoods = {}
p = mean / var
r = mean ** 2 / (var - mean)
likelihoods['nbinom'] = values.map(lambda val: scp.stats.nbinom.logpmf(val, r, p)).sum()
lam = mean
likelihoods['poisson'] = values.map(lambda val: scp.stats.poisson.logpmf(val, lam)).sum()
p = 1 / mean
print('P =', p)
likelihoods['geometric'] = values.map(lambda val: scp.stats.geom.logpmf(val, p)).sum()
best_fit = max(likelihoods, key=lambda x: likelihoods[x])
print("Best fit:", best_fit)
print("Likelihood:", likelihoods[best_fit])
print(likelihoods)

q = scp.stats.geom.rvs(p = p, loc = 0, size = len(delays))
plt.hist(q)
plt.savefig('D:\\MFK\\dist.jpg')
plt.clf()

# Задание 8.
posDelays = fixedFlights[fixedFlights['dep_delay'] > 0]
flightCounts = []
depDelayAvs = []
for month in range(1, 13):
    monthFlights = posDelays[posDelays['month'] == month]
    flightCounts.append(len(monthFlights))
    depDelayAvs.append(monthFlights['dep_delay'].describe()['mean'])
corr = np.corrcoef(depDelayAvs, flightCounts)[0, 1]
xmean = pd.Series(flightCounts).describe()['mean']
ymean = pd.Series(depDelayAvs).describe()['mean']
b = ((np.array(flightCounts) * np.array(depDelayAvs)).mean() - xmean * ymean) / (((np.array(flightCounts) ** 2).mean() - xmean ** 2))
a = ymean - b * xmean
print('Correlation coefficient = ', corr)
plt.scatter(flightCounts, depDelayAvs)
flightCounts = np.array(flightCounts)
flightCounts.sort()
print(flightCounts)
plt.plot(flightCounts, a + b * flightCounts)
plt.xlabel("Flight counts")
plt.ylabel("Average delay")
plt.savefig('D:\\MFK\\correlation.jpg')
plt.clf()
print('Regression equation:', b, '* x +', a)

# Задание 9.
depDelayAvs = []
for hour in range(24):
    timeFlights = fixedFlights[fixedFlights['dep_time'] % 100 == hour]
    depDelayAvs.append(timeFlights['dep_delay'].describe()['mean'])
plt.scatter(range(24), depDelayAvs)
plt.xlabel("Hour")
plt.ylabel("Average departure delay")
plt.savefig('D:\\MFK\\hours.jpg')
plt.clf()

counts = []
for hour in range(24):
    timeFlights = fixedFlights[fixedFlights['dep_time'] % 100 == hour]
    goodFlights = timeFlights[timeFlights['dep_delay'] > 0]
    counts.append(len(goodFlights) / len(timeFlights))
plt.scatter(range(24), counts)
plt.xlabel("Hour")
plt.ylabel("Share of positive delays")
plt.savefig('D:\\MFK\\posHours.jpg')
plt.clf()

# Задание 10.
firstClusterOld, secondClusterOld = clustering(companyMeans)
print('List of punctual companies:', firstClusterOld, sep = '\n')
print('List of non-punctual companies', secondClusterOld, sep = '\n')
distanceDesc = fixedFlights['distance'].describe()
shortDistanceFlights = fixedFlights[fixedFlights['distance'] < distanceDesc['mean']]
longDistanceFlights = fixedFlights[fixedFlights['distance'] > distanceDesc['mean']]
shortDistCompanyDelays = []
for company in shortDistanceFlights['carrier'].unique():
    desc = shortDistanceFlights[shortDistanceFlights['carrier'] == company]['dep_delay'].describe()
    shortDistCompanyDelays.append((desc['mean'], company))
longDistCompanyDelays = []
for company in longDistanceFlights['carrier'].unique():
    desc = longDistanceFlights[longDistanceFlights['carrier'] == company]['dep_delay'].describe()
    longDistCompanyDelays.append((desc['mean'], company))
firstClusterOld, secondClusterOld = clustering(shortDistCompanyDelays)
print('List of punctual companies on short flights:', firstClusterOld, sep = '\n')
print('List of non-punctual companies on short flights', secondClusterOld, sep = '\n')
firstClusterOld, secondClusterOld = clustering(longDistCompanyDelays)
print('List of punctual companies on long flights:', firstClusterOld, sep = '\n')
print('List of non-punctual companies on long flights', secondClusterOld, sep = '\n')

# Задание 12
f = open('D:\\MFK\\tailnumsDep.txt', 'w')
tails = fixedFlights['tailnum'].unique()
tailsDel = dict()
for tail in tails:
    tailsDel[tail] = fixedFlights[fixedFlights['tailnum'] == tail]['dep_delay'].values
for tail in tailsDel:
    if tail == 'N14228':
        continue
    res = scp.stats.ttest_ind(tailsDel['N14228'], tailsDel[tail], equal_var = False)
    if res[1] < 0.05:
        f.write(tail + ' ' + str(res[1]) + '\n')

f.close()
f = open('D:\\MFK\\tailnumsArr.txt', 'w')
tailsDel = dict()
for tail in tails:
    tailsDel[tail] = fixedFlights[fixedFlights['tailnum'] == tail]['arr_delay'].values
for tail in tailsDel:
    if tail == 'N14228':
        continue
    res = scp.stats.ttest_ind(tailsDel['N14228'], tailsDel[tail], equal_var = False)
    if res[1] < 0.05:
        f.write(tail + ' ' + str(res[1]) + '\n')
f.close()
