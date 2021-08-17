from random import randint
import matplotlib.pyplot as plt

k = 3
A = [2*i for i in range(k)]
C = [5*i for i in range(k)] #On a bien C[0]=0
#/!\ C[j] équivaut à C(j+1) dans le rapport /!\

def aveugle(Npred, N):
    """
    Algorithme aveugle avec prédictions
    
    Parameters
    ----------
    Npred : int
        Le nombre d'instant prédit entre les deux tâches.
        
    N : int
        Le nombre d'instant réél entre les deux tâches.
     
    Returns
    -------
    int
        Coût en terme d'énergie dépensée entre les deux tâches.

    """
    if Npred < C[1]/(A[0]-A[1]): #Aussi exprimé dans la boucle for suivante
        # Rester dans l'état 1 jusqu'à l'arrivée d'une nouvelle tâche pour un coût de A_1.N
        return A[0]*N
    
    for i in range(1, k):
        if Npred < (C[i]-C[i-1])/(A[i-1]-A[i]):
            # Passer dans l'état i dès le premier instant pour un coût de A_i.N + C_i
            return A[i-1]*N + C[i-1]
    # Sinon, c'est-à-dire si Npred >= (Ck-C(k-1))/(A(k-1)-Ak), passer dans l'état k dès le 
    # premier instant pour un coût de A_k.N + C_k
    return A[k-1]*N +C[k-1]

def simpleRobCons(Npred, lambd, N):
    """
    Algorithme aveugle avec prédictions
    
    Parameters
    ----------
    Npred : int
        Le nombre d'instant prédit entre les deux tâches.
     
    lambd : float
            L'hyperparamètre de robustesse.
            
    N : int
        Le nombre d'instant réél entre les deux tâches.
     
    Returns
    -------
    int
        Coût en terme d'énergie dépensée entre les deux tâches.

    """
    if Npred < C[1]/(A[0]-A[1]):
        # Passer dans l'état 2 au temps C1/(lambda(A1-A2)) puis dans l'état i au temps 
        # (Ci-C(i-1))/(lambda(A(i-1)-Ai))
        if N < C[1]/(lambd*(A[0]-A[1])): #Aussi exprimé dans la boucle for suivante
            # On n'a pas le temps de passer à l'état 2
            return A[0]*N
        
        for i in range(1, k-1):
            if N < (C[i]-C[i-1])/(lambd*(A[i-1]-A[i])):
                # On n'a pas le temps de passer à l'état i+1
                res = 0
                tmp = 0
                for j in range(1, i):
                    res += A[j-1]*((C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1-tmp)
                    tmp = (C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1  
                return res + A[i-1]*(N-tmp) + C[i-1]
        
    for i in range(2, k):
        if Npred < (C[i]-C[i-1])/(A[i-1]-A[i]):
        # Pour tout l<=i, passer dans l'état l au temps lambda*(Cl-C(l-1))/((A(l-1)-Al)), puis pour
        # tout r>i, passer dans l'état r au temps (Cr-C(r-1))/(lambda(A(r-1)-Ar))
            for l in range(1, i+1):
                if N < lambd*(C[l]-C[l-1])/((A[l-1]-A[l])):
                    # On n'a pas le temps de passer à l'état l+1
                    res = 0
                    tmp = 0
                    for j in range(1, l):
                        res += A[j-1]*(lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1-tmp)
                        tmp = lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1  
                    return res + A[l-1]*(N-tmp) + C[l-1]
             
            # Si on arrive ici, on prend déjà en compte les coûts pour les changement d'états l<=i    
            res = 0
            tmp = 0
            for j in range(1, i+1): #PB pour i=1 ne devrait pas aller la, et i=2 il y a 2fois lambd au num
                res += A[j-1]*(lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1-tmp)
                tmp = lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1 
                
            for r in range(i+1, k-1): #PB fin a k+1 ou k-1 ? VERIF tous les intervalles
                if N < (C[r]-C[r-1])/(lambd*(A[r-1]-A[r])):
                    # On n'a pas le temps de passer à l'état r+1
                    for j in range(1, r):
                        res += A[j-1]*((C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1-tmp)
                        tmp = (C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1  
                    return res + A[r-1]*(N-tmp) + C[r-1]
                                     
    # Sinon, c'est-à-dire si Npred >= (Ck-C(k-1))/(A(k-1)-Ak), passer dans l'état i au temps 
    # lambda(Ci-C(i-1))/(A(i-1)-Ai)
    for i in range(1, k-1): #PB fin a k+1 ou k-1 ? VERIF tous les intervalles
        if N < lambd*(C[i]-C[i-1])/(A[i-1]-A[i]):
            # On n'a pas le temps de passer à l'état i+1
            res = 0
            tmp = 0
            for j in range(1, i):
                res += A[j-1]*(lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1-tmp)
                tmp = lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1  
            return res + A[i-1]*(N-tmp) + C[i-1] 
        
    # Sinon, c'est-à-dire si N >= lambda(Ck-C(k-1))/(A(k-1)-Ak)
    res = 0
    tmp = 0
    for j in range(1, k):
        res += A[j-1]*(lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1-tmp)
        tmp = lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1     
    return res + A[k-1]*(N-tmp) + C[k-1]
        
def erreur_prediction(Npred, N):
    """ 
    Calcule l'erreur de prédiction
    
    Parameters
    ----------
    Npred : int
           La prédiction de la durée entre les tâches
           
    N : int
           La vraie durée séparant les deux tâches 
           
    Returns
    -------
    float 
        L'erreur de prédiction
    """
    return abs(Npred - N)

def show_multiple_sims(XYlegendes):
    """ 
    Affiche le graphique demandé
    
    Parameters
    ----------
    XYlegendes : list[(list[float], list[float], str)]
           Une liste composée de tuple dont le premier élément est une liste des X (échantilloné à intervalle régulier), 
           le second une liste des Y (ratio de compétitivité) et le dernier le nom pour la légende     
    """
    for _ in XYlegendes:
        X, Y, legende = _
        plt.plot([X[s] for s in range(len(X[:-1])) if Y[s] != 0], [y for y in Y if y != 0] , label=legende)
        #plt.plot(X, Y, label=legende)

        plt.xlabel('Erreur de prédiction')
        plt.ylabel('Rapport de compétitivité')
        #plt.legend(loc='best')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    
def simulations():
    """
    Permet de faire une simulations des algorithmes précédents
    
    Parameters
    ----------
    Npred : int
            Le nombre prédit d'instants entre les tâches.
            
    lambd : float
            L'hyperparamètre de robustesse.
            
    N : int
            Le vrai nombre d'instants entre les tâches.

    Returns
    -------
    int
        C PB

    """
    # Tirage aléatoire de N et de Npred

    nbExemples = 15000 #Nombre d'exemples du problème
    Nmax = 10*k #Nombre d'instant entre les tâches maximum
    # En fonction du nombre d'état pour avoir potentiellement le temps de parcourir tous les états
    
    Ns = [randint(1, Nmax) for i in range(nbExemples)]
    Npreds = [Ns[i] + randint(0, Nmax-Ns[i]) - randint(0, Ns[i]-1)  for i in range(nbExemples)]
    #print("Ns = ", Ns)
    #print("Npreds = ", Npreds)
    
    Ns_Npred = [(Ns[i], Npreds[i]) for i in range(nbExemples)]
    Ns_Npred.sort(key=lambda vpa: erreur_prediction(vpa[1], vpa[0]))
    
    lambd = 0.5 #PB a faire varier
    # Définition des points du graphique
    X = [erreur_prediction(p, v) for (v, p) in Ns_Npred]
    #Ratio de compétitibité déterminé par algo / aveugle(N,N) puisque aveugle(N,N) est l'optimum
    Yaveugle = [aveugle(Npred, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons05 = [simpleRobCons(Npred, lambd, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons0 = [simpleRobCons(Npred, lambd, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons1 = [simpleRobCons(Npred, lambd, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    
    # À partir d'ici, on crée des classes de valeurs comme pour un histogramme, dont on tirera le maximum, 
    # puisque le rapport de compétitivité est calculé en fonction du pire cas des algorithmes
   
    def locale(X, Y, legend):
        """
        Permet de créer des classes de valeurs comme pour un histogramme, dont on tirera le maximum
        
        Parameters
        ----------
        X : list[float]
            Liste des erreurs de prédictions.
                
        Y : list[float]
            Liste des ratios de compétitivité.
                
        legend : str
            Nom de la légende.
    
        Returns
        -------
        (list[float], list[float], str)
            Un tuple contenant la liste des X et des Y du graphique et la légende associée.

    """
    # precision : int, nombre de classes voulues. Si trop élevé, peut avoir un impact néfaste (classes vides).
        precision = 100
        xmin = min(X)
        xmax = max(X)
        step = (xmax - xmin )/precision
        stages = [xmin]
        while max(stages) < xmax:
            stages.append(stages[-1] + step)
        classes = [[Y[j] for j in range(len(X)) if X[j] >= stages[i] and X[j] < stages[i+1]] for i in range(len(stages) - 1)]
        
        # On met un 0 pour les classes vides. Les fonctions d'affichage devront en tenir compte.
        newY = [max(c) if len(c) > 0 else 0 for c in classes]
        return (stages, newY, legend)
    
    resA = []
    # Algorithme aveugle
    resA.append(locale(X, Yaveugle, "Aveugle"))
    # Algorithme simple robuste et consistant pour un lambda de 0.5
    resA.append(locale(X, YsimpleRobCons05, "Simple Robuste Consistant lamb = 0.5"))
    # Algorithme simple robuste et consistant pour un lambda de 0
    resA.append(locale(X, YsimpleRobCons0, "Simple Robuste Consistant lamb = 0"))
    # Algorithme simple robuste et consistant pour un lambda de 1
    resA.append(locale(X, YsimpleRobCons1, "Simple Robuste Consistant lamb = 1"))
    
    # affichage du graphique
    show_multiple_sims(resA)
    


simulations()