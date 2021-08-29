#-------改过的不要cut以前的
def Ell_new(Q):
    lmax=1024
    cut=10
    ell = np.arange(0.,lmax+1)
    smallell = ell[ell<=cut]; largeell=ell[ell>cut]
    largex=np.logspace(np.log10(largeell[0]), np.log10(largeell[-1]), Q+1)
    largex_int = np.ones(len(largex))
    for i in range(len(largex_int)):
        largex_int[i] = round(largex[i],0)
    new_ell=[]
    for i in range(len(largex)-1):
        new_ell.append( round((largex[i]+largex[i+1]) / 2, 0) )
    ell_new = np.hstack(new_ell)
    Nbin = len(ell_new)
    print 'bin edge = ', largex_int
    print 'ell_new = ', ell_new
    print '===> Number of bins =',len(ell_new)
    return largex_int, ell_new, lmax


# -----改过的，cut之前的不要
def C_new(cl,Q):
    largex_int, ell_new, lmax = Ell_new(Q)
    ell = np.arange(0.,lmax+1)
    bin_averages=[]
    for i in range(len(largex_int)-1):
        bin_averages.append(np.mean(cl[int(largex_int[i]):int(largex_int[i+1])]))
    C_new = bin_averages * ell_new*(ell_new+1)/(2.*np.pi) 
    return ell_new, C_new
def C_new_noco(cl,Q):
    largex_int, ell_new, lmax = Ell_new(Q)
    ell = np.arange(0.,lmax+1)
    bin_averages=[]
    for i in range(len(largex_int)-1):
        bin_averages.append(np.mean(cl[int(largex_int[i]):int(largex_int[i+1])]))
    return ell_new, bin_averages

def delta_ell(Q):
    largex_int, ell_new, lmax = Ell_new(Q)
    delta_largex_int=[]
    for i in range(len(largex_int)-1):
        delta_largex_int.append( largex_int[i+1] - largex_int[i] )
        delta_ell = np.hstack(delta_largex_int)
    print 'delta_ell = ',delta_ell
    return delta_ell
# -------------------here the clkg_err didn't divided by sqrt(delta_ell*(2ell+1))
def C_err(clkg_err,Q):
    largex_int, ell_new, lmax = Ell_new(Q)
    delta_l = delta_ell()
    err = C_new_noco(clkg_err) / np.sqrt( delta_l *(2.*ell_new+1) )
    return err

def bin_l(cl):
    bin_averages = []
    for l in range(L):
        cl[l] = l*(l+1)/2/np.pi*(cl[l])    
    for q in range(Q):
        bin_averages.append(sum(cl[q*L/Q:((q+1)*L/Q)]/(L/Q)))
    return bin_averages
def bin_l_noco(cl):
    bin_averages = []
    for l in range(L):
        cl[l] = (cl[l])    
    for q in range(Q):
        bin_averages.append(sum(cl[q*L/Q:((q+1)*L/Q)]/(L/Q)))
    return bin_averages
def ell_lin(D):
    L=1024
    ell_lin = np.ones(D)
    for q in range(D):
        ell_lin[q] = (2*q+1)*L/D/2
    print ell_lin
    return ell_lin