

def boosting(Aq, lamda, mu, eta):
    if Aq <= lamda:
        aq = 1
    elif Aq >= lamda + mu:
        aq = -1
    else:
        aq=1-2 * (Aq - lamda) / mu
        
    return aq