import sys 
sys.path.append('/data/s6/zysun/code/')
from C_new import C_new, C_new_noco

def get_covariance(Nsam):
    loc = 'cmb-cross-correlation/cls_cut_new/err_kg'
    cls_kg_mtd2 = np.load('%s/cls_kg_method2_Nsam%s_%sto%s_cut%s.npy'%(loc,Nsam,zmin,zmax,cut))

    c=cls_kg_mtd2
    
    c_new = np.ones((4, Nsam,len(ell_new)))
    cov_mtd2 = np.ones((4, len(ell_new), len(ell_new)))
    r_mtd2 = np.ones((4, len(ell_new), len(ell_new)))
    r_mtd2_minus_diag = np.ones((4, len(ell_new), len(ell_new)))
    for i in range(4):
        for n in range(Nsam):
            c_new[i,n] = C_new_noco(c[i,n])
# np.cov
# bias:默认为False,此时标准化时除以n-1；反之为n。其中n为观测数
# ddof:类型是int，当其值非None时，bias参数作用将失效。当ddof=1时，将会返回无偏估计（除以n-1），即使指定了fweights和aweights参数；当ddof=0时，则返回简单平均值。
        cov_mtd2[i] = np.cov(c_new[i], rowvar=False)  
        r_mtd2[i] = np.corrcoef(c_new[i], rowvar=False)
        r_mtd2_minus_diag[i] = r_mtd2[i]-np.diag(np.diag(r_mtd2[i]))
    np.save('%s/cov_mat_mtd2_Nsam%s_cut%s.npy'%(loc,Nsam,cut),cov_mtd2)
    return cov_mtd2, r_mtd2

def plot_cov_matrix(Nsam):
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
   
    for i in range(4):
        print '%1.1f'%np.max(abs(r_mtd2_minus_diag[i]*100))
    fig,axs = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(15,10),sharey=False)
    for ax,i in zip(axs.flat,range(4)):
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        r= ax.imshow(r_mtd2[i], cmap='coolwarm', origin='lower', extent=(0,10,0,10) )
        ax.text(6.5,1,'%1.1f'%np.max(abs(r_mtd2_minus_diag[i]*100))+'%', color='gold',ha='left', va= 'bottom',fontsize=24,fontstyle='normal',fontweight='semibold')

        ax.tick_params(labelsize=16)
        ax.set_xlabel('Multipole bin', fontsize = 22)
        ax.set_ylabel('Multipole bin', fontsize = 24)
        ax.set_title('$%s<z<%s, N_{random}=%s$'%(zmin, zmax, Nsam), fontsize=16)
        xmajorLocator = MultipleLocator(2) #将x主刻度标签设置为20的倍数
        xmajorFormatter = FormatStrFormatter('%1.0f') #设置x轴标签文本的格式
        xminorLocator = MultipleLocator(1) #将x轴次刻度标签设置为5的倍数
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_major_formatter(xmajorFormatter)

    cb1 = fig.colorbar(r,ax=axs[0,0])
    cb2 = fig.colorbar(r,ax=axs[0,1])
    cb3 = fig.colorbar(r,ax=axs[1,0])
    cb4 = fig.colorbar(r,ax=axs[1,1])
    cb1.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
    cb2.ax.tick_params(labelsize=16)  
    cb3.ax.tick_params(labelsize=16)  
    cb4.ax.tick_params(labelsize=16)  
plot_cov_matrix(100)
plot_cov_matrix(200)
plot_cov_matrix(300)

# plt.savefig('figure/cov_mtd2_300.png',dpi=300,bbox_inches='tight')

