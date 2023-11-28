
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error
from scipy.special import spence
from boosting import boosting

import queue
import random

def get_unique_random_number(small_numbers,medium_numbers,large_numbers,small_numbers_set,medium_numbers_set,large_numbers_set):
    while True:
        random_queue = random.choice([small_numbers, medium_numbers, large_numbers])
        if random_queue is small_numbers and not small_numbers.empty():
            number = small_numbers.get()
            small_numbers_set.remove(number)
            return number
        elif random_queue is medium_numbers and not medium_numbers.empty():
            number = medium_numbers.get()
            medium_numbers_set.remove(number)
            return number
        elif random_queue is large_numbers and not large_numbers.empty():
            number = large_numbers.get()
            large_numbers_set.remove(number)
            return number



def kldivergence(arra,arrb):
    div = np.sum(np.log(arra / arrb) * arra)
    return div


# Generate one sample
def synoposis_generator_pseu(n):
    p = [1/n] * n 
    sample = np.random.multinomial(n, p)
    synopsis = [x / sum(sample) for x in sample]
    return synopsis



def random_budget_allocation(epsilon,mu,small_numbers,medium_numbers,large_numbers,small_numbers_set,medium_numbers_set,large_numbers_set):
    a = 1
    
    i = get_unique_random_number(small_numbers,medium_numbers,large_numbers,small_numbers_set,medium_numbers_set,large_numbers_set)

    C=2/np.abs(a)/mu
    
    m_square=spence((np.pi**2/6)-epsilon/C)**(-1)

    eta=(np.e**((1-m_square**(2*i))/i/i/np.abs(a)) -1)/(np.e**((1-m_square**(2*i))/i/i/np.abs(a)) +1)
    
    return eta

def framework(epsilon, tslot, DEFAULT_DIST_LEN,lamda, mu,small_numbers,medium_numbers,large_numbers,small_numbers_set,medium_numbers_set,large_numbers_set):
    difference=[]
    sigma=2
    length = 6500 ## Define the synopsis pool size
    query_number = 2
    sampler_distribution_query = [[1/length] * length for _ in range(query_number)]
    queryset = []
    output_ds1=[]
    output_ks1=[]
    total_accum = np.zeros(DEFAULT_DIST_LEN)
    query = 1

    for t in range(1,int(tslot)):

        ## Generate data
        current_slot = np.random.normal(DEFAULT_DIST_LEN,sigma,DEFAULT_DIST_LEN)
        current_slot[current_slot < 0] = 0
        total_accum += current_slot
        current_slot /= np.sum(current_slot)


        ## Get budget
        eta = random_budget_allocation(epsilon,mu,small_numbers,medium_numbers,large_numbers,small_numbers_set,medium_numbers_set,large_numbers_set)
        total_accum[total_accum < 0] = 0
        current_slot=np.array(current_slot)
        current_slot[current_slot < 0] = 0
        divs_acc = total_accum
        divs_acc /= divs_acc.sum() #true answers pdf
        divs_cur = np.array(current_slot)
        divs_cur /= divs_cur.sum()
        queryset.append(divs_acc)
        queryset.append(divs_cur)
        output = np.zeros(len(divs_acc))
        alpha=1/2*np.log((1+2*eta)/(1-2*eta))

        synoposislist = [None] * length
        possible_outcomes = [i for i in range(length)]
        index_list=np.random.choice(possible_outcomes, 20, p=sampler_distribution_query[query]) ###### defined by domain size
        for i in index_list:

            structure=synoposis_generator_pseu(DEFAULT_DIST_LEN)
            output = [output[m]+structure[m] for m in range(len(structure))]
            while(structure in synoposislist):
                structure=synoposis_generator_pseu(DEFAULT_DIST_LEN)


            synoposislist[i]=structure
            l1_dist = np.linalg.norm(structure - divs_acc, ord=1)

            sampler_distribution_query[query][i]=boosting(l1_dist,lamda,mu,eta)

        uq1t=np.exp(alpha*np.sum(sampler_distribution_query[query]))
        for i in range(len(sampler_distribution_query[query])):
            sampler_distribution_query[query][i]=sampler_distribution_query[query][i]*uq1t
        sampler_distribution_query[query] = [x if x > 0 else 0.0001 for x in sampler_distribution_query[query]]
        sampler_distribution_query[query] /= np.sum(sampler_distribution_query[query])
        output /= np.sum(output)

        output_ds1.append(mean_squared_error(output, queryset[query]))
        output_ks1.append(entropy(queryset[query], output))


    return np.mean(output_ds1),np.mean(output_ks1)





def main():
    #sensitivity = 2
    lamda=0.5
    mu=0.5
    tslot=10
    DEFAULT_DIST_LEN = 100
    epsilon = 2
    # pritvate_result=framework(epsilon, tslot, DEFAULT_DIST_LEN,lamda, mu, small_numbers,medium_numbers,large_numbers,small_numbers_set,medium_numbers_set,large_numbers_set)


    # Initialize queues and sets
    small_numbers = queue.Queue()
    medium_numbers = queue.Queue()
    large_numbers = queue.Queue()

    small_numbers_set = set()
    medium_numbers_set = set()
    large_numbers_set = set()

    # Define the ranges for each category
    small_range = range(1, 10000)
    medium_range = range(10000, 20000)
    large_range = range(20000, 30000)
    # Fill each queue with its respective range and also add it to the set
    for i in small_range:
        small_numbers.put(i)
        small_numbers_set.add(i)

    for i in medium_range:
        medium_numbers.put(i)
        medium_numbers_set.add(i)

    for i in large_range:
        large_numbers.put(i)
        large_numbers_set.add(i)

    # Fill each queue with its respective range and also add it to the set
    for i in small_range:
        small_numbers.put(i)
        small_numbers_set.add(i)

    for i in medium_range:
        medium_numbers.put(i)
        medium_numbers_set.add(i)

    for i in large_range:
        large_numbers.put(i)
        large_numbers_set.add(i)


    pritvate_result=framework(epsilon, tslot, DEFAULT_DIST_LEN,lamda, mu, small_numbers,medium_numbers,large_numbers,small_numbers_set,medium_numbers_set,large_numbers_set)
    print(pritvate_result)
if __name__ == "__main__":
    main()