# kolam_generator.py

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit

# Set random seed for reproducibility
np.random.seed(seed=1)

class KolamDraw(object):
    
    def __init__(self,ND):
        # Initialization GitHub Version
        # ND - Kolam dimension odd integer > 3
        # Nx - Extended kolam dimension
        # A1 - Gate matrix
        # F1 - Occupancy companion matrix
        # Ns - Maximum number of kolam segments possible in a one-stroke kolam
        
        self.ND = ND
        self.Nx = ND + 1
        self.A1 = np.ones((self.Nx,self.Nx))*99
        self.F1 = np.ones((self.Nx,self.Nx))
        self.Ns = 2*(self.ND**2 + 1) + 5
        self.boundary_type = 'diamond'

    def set_boundary(self, boundary_type='diamond'):
        self.boundary_type = boundary_type

    def is_inside_boundary(self, i, j):
        ND = self.ND
        x = i - ND
        y = j - ND
        if self.boundary_type == 'diamond':
            return True
        elif self.boundary_type == 'corners':
            threshold = ND * 0.3
            return (abs(x) >= threshold) and (abs(y) >= threshold)
        elif self.boundary_type == 'fish':
            dist = np.sqrt(x**2 + y**2)
            ring_width = ND * 0.35
            ring_num = int(dist / ring_width)
            return (ring_num % 2) == 0
        elif self.boundary_type == 'waves':
            freq = 2 * np.pi / (ND * 0.6)
            wave1 = np.sin(x * freq)
            wave2 = np.sin(y * freq)
            interference = wave1 + wave2
            return interference > 0
        elif self.boundary_type == 'fractal':
            xi = int(abs(x) + ND)
            yi = int(abs(y) + ND)
            return (xi & yi) % 4 < 2
        elif self.boundary_type == 'organic':
            dist = np.sqrt(x**2 + y**2)
            angle = np.arctan2(y, x)
            wobble = np.sin(angle * 3) * ND * 0.2
            max_dist = ND * 0.8 + wobble
            return dist < max_dist
        return True

    def ResetGateMatrix(self):
        # Resets both gate matrix and the occupancy companion matrix
        # Establishes kolam boundaries
        
        Nx2 = int(self.Nx/2)
        Nx1 = self.Nx - 1
        A = self.A1*1
        F = self.F1*1
        
        for i in range(self.Nx):
            A[0,i] = A[i,0] = A[self.Nx-1,i] = A[i,self.Nx-1] = 0
            F[0,i] = F[i,0] = F[self.Nx-1,i] = F[i,self.Nx-1] = 0
        for i in range(1,self.Nx-1):
            A[i,i] = A[i,self.Nx-1-i] = 1
            F[i,i] = F[i,self.Nx-1-i] = 0
        # Apply boundary mask if not diamond
        if self.boundary_type != 'diamond':
            for i in range(self.Nx):
                for j in range(self.Nx):
                    if not self.is_inside_boundary(i, j):
                        A[i,j] = 0
                        F[i,j] = 0
            
        return(A, F)

    def toss(self,bias):
        # Returns a biased random output
        # Called out in AssignGates
        
        x = np.random.randint(0,1000)/1000
        if x > bias:
            return (1)
        else:
            return (0)

    def AssignGates(self,krRef,Kp,Ki):
        # Assigns random gate distribution
        # Called out in Dice
        
        A, F = self.ResetGateMatrix()
        Asum = A.sum()
        Fsum = F.sum()
        
        errAckr = 0.0
        count1 = 0
        count01 = 1
        Nx1 = self.Nx-1
        Nx2 = int(self.Nx/2)
        
        for i in range (1,Nx2):
            for j in range (i,self.Nx-i):
                errkr = krRef - (count1/count01)
                errAckr = errAckr + errkr
                kr = krRef + Kp*errkr + Ki*errAckr
                
                if F[i,j] == 1:
                    if F[i,j+1] == 1:
                        A[i,j] = A[j,i] = A[Nx1-i,Nx1-j] = A[Nx1-j,Nx1-i] = self.toss(kr)
                        F[i,j] = F[j,i] = F[Nx1-i,Nx1-j] = F[Nx1-j,Nx1-i] = 0
                        count01 = count01 + 4
                        if A[i,j] > 0.9:
                            count1 = count1 + 4
                        if (A[i-1,j] + A[i-1,j+1] + A[i,j]) < 0.1:
                            x = 1
                        else:
                            x = self.toss(kr)
                        A[i,j+1] = A[j+1,i] = A[Nx1-i,Nx1-1-j] = A[Nx1-1-j,Nx1-i] = x
                        F[i,j+1] = F[j+1,i] = F[Nx1-i,Nx1-1-j] = F[Nx1-1-j,Nx1-i] = 0
                        count01 = count01 + 4
                        if A[i,j+1] > 0.9:
                            count1 = count1 + 4
                    if F[i,j+1] <= 0:
                        if (A[i-1,j] + A[i-1,j+1] + A[i,j+1]) < 0.1:
                            x = 1
                        else:
                            x = self.toss(kr)
                        A[i,j] = A[j,i] = A[Nx1-i,Nx1-j] = A[Nx1-j,Nx1-i] = x
                        F[i,j] = F[j,i] = F[Nx1-i,Nx1-j] = F[Nx1-j,Nx1-i] = 0
                        count01 = count01 + 4
                        if A[i,j] > 0.9:
                            count1 = count1 + 4
                
                if F[i,j] <= 0:
                    if F[i,j+1] == 1:
                        if (A[i-1,j] + A[i-1,j+1] + A[i,j]) < 0.1:
                            x = 1
                        else:
                            x = self.toss(kr)
                        A[i,j+1] = A[j+1,i] = A[Nx1-i,Nx1-1-j] = A[Nx1-1-j,Nx1-i] = x
                        F[i,j+1] = F[j+1,i] = F[Nx1-i,Nx1-1-j] = F[Nx1-1-j,Nx1-i] = 0
                        count01 = count01 + 4
                        if A[i,j+1] > 0.9:
                            count1 = count1 + 4
        
        return(A, F, kr)

    def NextStep(self,icg,jcg,ce):
        # Generates the next kolam evolution path given a current coordinate and a entry direction
        
        icgx = icg + self.ND
        jcx = jcg + self.ND
        icgx2 = int(np.floor(icgx/2))
        jcx2 = int(np.floor(jcx/2))
        
        calpha = np.mod(ce,2)
        if ce>1:
            cbeta = -1
        else:
            cbeta = 1
        if np.mod(int(icgx + jcx),4) == 0:
            cgamma = -1
        else:
            cgamma = 1
            
        if self.A[icgx2,jcx2]>0.5:
            cg = 1
        else:
            cg = 0
            
        cgd = 1-cg
        calphad = 1-calpha
        nalpha = cg*calpha + cgd*calphad
        nbeta = (cg + cgd*cgamma)*cbeta
        nh = (calphad*cgamma*cgd + calpha*cg)*cbeta
        nv = (calpha*cgamma*cgd + calphad*cg)*cbeta
        ing = int(icg + nh*2)
        jng = int(jcg + nv*2)
        ingp = icg + cgd*(calphad*cgamma - calpha)*cbeta*0.5
        jngp = jcg + cgd*(calpha*cgamma - calphad)*cbeta*0.5
        
        if nalpha == 0:
            if nbeta == 1:
                ne = 0
            else:
                ne = 2
        if nalpha == 1:
            if nbeta == 1:
                ne = 1
            else:
                ne = 3
                
        return(ing,jng,ne,ingp,jngp)

    def XNextSteps(self,icgo,jcgo,ceo,Ns):
        # Generates the next Ns steps of the kolam evolution path
        
        ijcx = np.zeros((Ns,2))
        cex = np.zeros((Ns))
        ijcp = np.zeros((Ns,2))
        ijcx[0,:] = [icgo,jcgo]
        cex[0] = ceo
        
        for i in range(Ns-1):
            ijcx[i+1,0], ijcx[i+1,1], cex[i+1], ijcp[i,0], ijcp[i,1] = self.NextStep(
                ijcx[i,0], ijcx[i,1], cex[i])
        return (ijcx, cex, ijcp)

    def PathCount(self):
        # Counts the number of evolving kolam segments from a fixed starting coordinate
        
        Ns = self.Ns
        ijcx = np.zeros((Ns,2))
        cex = np.zeros((Ns))
        ijcp = np.zeros((Ns,2))
        cex[0] = 0
        Flag1 = Flag2 = 0
        isx = 0
        isa = 0
        
        ijcx[0,0] = 2*np.random.randint(0,2) - 1
        ijcx[0,1] = 2*np.random.randint(0,2) - 1
        
        cex[0] = 0
        
        while isa < Ns-2:
            
            isa = isa + 1
            
            ijcx[isa,0], ijcx[isa,1], cex[isa], ijcp[isa-1,0], ijcp[isa-1,1] = self.NextStep(ijcx[isa-1,0], ijcx[isa-1,1], cex[isa-1])
            if int(ijcx[isa,0]) == int(ijcx[0,0]):
                if int(ijcx[isa,1]) == int(ijcx[0,1]):
                    if int(cex[isa]) == int(cex[0]):
                        Flag1 = 1
                        isx = isa
                        isa = Ns - 1
        
        return(isx)

    def Dice(self,krRef,Kp,Ki,Nthr):
        # Performs several random gate distributions and selects the one with the longest kolam path evolution
        
        Ns = self.Ns
        ijcx = np.zeros((Ns,2))
        cex = np.zeros((Ns))
        ijcp = np.zeros((Ns,2))
        cex[0] = 0
        krx = np.zeros((Nthr))
        ith = int(0)
        ithx = int(0)
        ismax = int(0)
        
        while ith < Nthr:
            
            self.A, self.F, krx[ith] = self.AssignGates(krRef,Kp,Ki)
            Asum = self.A.sum()
            Fsum = self.F.sum()
            
            Flag1 = Flag2 = 0
            isx = 0
            isa = 0
            ijcx[0,0] = ijcx[0,1] = 1
            cex[0] = 0
            
            while isa < Ns-2:
                
                isa = isa + 1
                
                ijcx[isa,0], ijcx[isa,1], cex[isa], ijcp[isa-1,0], ijcp[isa-1,1] = self.NextStep(ijcx[isa-1,0], ijcx[isa-1,1], cex[isa-1])
                if int(ijcx[isa,0]) == int(ijcx[0,0]):
                    if int(ijcx[isa,1]) == int(ijcx[0,1]):
                        if int(cex[isa]) == int(cex[0]):
                            Flag1 = 1
                            isx = isa
                            isa = Ns - 1
            
            if Flag1 == 1:
                if isx < Ns + 2:
                    Flag2 = 1
                if isx > ismax:
                    ismax = isx
                    Amax = self.A
                if isx > (Ns/2) - 2:
                    Flag2 = 2
                    ithx = ith
                    ith = int(Nthr + 1)
            
            ith = ith + 1
        
        return (self.A, self.F, Amax, isx, ithx, ismax, Flag1, Flag2, krx)

    def SwitchGate(self, ig, jg):
        # Performs a single gate switching
        
        Flag = 0
        Ax = self.A*1
        Fx = self.F*1
        Nx = self.Nx
        if Ax[ig,jg] < 0.1:
            Ax[ig,jg] = Ax[jg,ig] = Ax[Nx-ig-1,Nx-jg-1] = Ax[Nx-jg-1,Nx-ig-1] = 1
            Fx[ig,jg] = Fx[jg,ig] = Fx[Nx-ig-1,Nx-jg-1] = Fx[Nx-jg-1,Nx-ig-1] = 0
            Flag = 1
        if Flag == 0:
            if Ax[ig,jg] > 0.9:
                Ax[ig,jg] = Ax[jg,ig] = Ax[Nx-ig-1,Nx-jg-1] = Ax[Nx-jg-1,Nx-ig-1] = 0
                Fx[ig,jg] = Fx[jg,ig] = Fx[Nx-ig-1,Nx-jg-1] = Fx[Nx-jg-1,Nx-ig-1] = 0
                Flag = -1
        
        return(Ax, Fx, Flag)

    def FlipTestSwitch(self, ksh, iL, iH):
        # Performs multiple flip test and switch action
        
        Ncx = self.PathCount()
        Ns = self.Ns
        Nx2 = int(self.Nx/2)
        
        iLx = np.maximum(np.minimum(iL,Nx2),1)
        iHx = np.maximum(np.minimum(iH,Nx2),iLx)
        for ig in range (iLx,iHx):
            for jg in range (ig,self.Nx-1-ig):
                
                Ax = self.A
                Fx = self.F
                
                if self.F[ig,jg] >= 0:
                    if self.toss(ksh) == 1:
                        self.A, self.F, Flag = self.SwitchGate(ig,jg)
                        Nc = self.PathCount()
                        
                        if Nc < Ncx:
                            self.A = Ax
                            self.F = Fx
                        if Nc > Ncx:
                            Ncx = Nc
                        if Ncx >= Ns-5:
                            break
        
        return(Ncx)

    def IterFlipTestSwitch(self, ksh, Niter, iL, iH):
        # Iterate flip test and switch actions
        Ns = self.Ns
        Ncx = self.PathCount()
        if Ncx < Ns:
            for iter in range (Niter):
                Ncx = self.FlipTestSwitch(ksh, iL, iH)
                if Ncx >= Ns-5:
                    break
        return (Ncx, self.A, self.F)


def plotkolam(ijngp, kolam_color='#1f77b4', theme='light'):
    """Plot the kolam pattern with color and theme"""
    Ncx = np.shape(ijngp)[0]
    if theme.lower() == 'dark':
        bg_color = '#1a1a1a'
    else:
        bg_color = 'white'
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ijngpx = (ijngp[:,0] + ijngp[:,1])/2
    ijngpy = (ijngp[:,0] - ijngp[:,1])/2
    ax.plot(ijngpx[0:Ncx-1], ijngpy[0:Ncx-1], color=kolam_color, linewidth=2)
    ND = int(np.max(np.abs(ijngp))) + 2
    Mn = -(ND+1)
    Mx = ND+1
    ax.set_xlim(Mn, Mx)
    ax.set_ylim(Mn, Mx)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig


# Complete one-stroke implementation (ported from attached file)
class CompleteKolamDraw(object):
    def __init__(self, ND):
        self.ND = ND
        self.Nx = ND + 1
        self.A1 = np.ones((self.Nx, self.Nx)) * 99
        self.F1 = np.ones((self.Nx, self.Nx))
        self.Ns = 2 * (self.ND**2 + 1) + 5

    def ResetGateMatrix(self):
        Nx2 = int(self.Nx / 2)
        Nx1 = self.Nx - 1
        A = self.A1 * 1
        F = self.F1 * 1
        for i in range(self.Nx):
            A[0, i] = A[i, 0] = A[self.Nx - 1, i] = A[i, self.Nx - 1] = 0
            F[0, i] = F[i, 0] = F[self.Nx - 1, i] = F[i, self.Nx - 1] = 0
        for i in range(1, self.Nx - 1):
            A[i, i] = A[i, self.Nx - 1 - i] = 1
            F[i, i] = F[i, self.Nx - 1 - i] = 0
        return (A, F)

    def toss(self, bias):
        x = np.random.randint(0, 1000) / 1000
        return 1 if x > bias else 0

    def AssignGates(self, krRef, Kp, Ki):
        A, F = self.ResetGateMatrix()
        errAckr = 0.0
        count1 = 0
        count01 = 1
        Nx1 = self.Nx - 1
        Nx2 = int(self.Nx / 2)
        for i in range(1, Nx2):
            for j in range(i, self.Nx - i):
                errkr = krRef - (count1 / count01)
                errAckr = errAckr + errkr
                kr = krRef + Kp * errkr + Ki * errAckr
                if F[i, j] == 1:
                    if F[i, j + 1] == 1:
                        A[i, j] = A[j, i] = A[Nx1 - i, Nx1 - j] = A[Nx1 - j, Nx1 - i] = self.toss(kr)
                        F[i, j] = F[j, i] = F[Nx1 - i, Nx1 - j] = F[Nx1 - j, Nx1 - i] = 0
                        count01 = count01 + 4
                        if A[i, j] > 0.9:
                            count1 = count1 + 4
                        if (A[i - 1, j] + A[i - 1, j + 1] + A[i, j]) < 0.1:
                            x = 1
                        else:
                            x = self.toss(kr)
                        A[i, j + 1] = A[j + 1, i] = A[Nx1 - i, Nx1 - 1 - j] = A[Nx1 - 1 - j, Nx1 - i] = x
                        F[i, j + 1] = F[j + 1, i] = F[Nx1 - i, Nx1 - 1 - j] = F[Nx1 - 1 - j, Nx1 - i] = 0
                        count01 = count01 + 4
                        if A[i, j + 1] > 0.9:
                            count1 = count1 + 4
                    if F[i, j + 1] <= 0:
                        if (A[i - 1, j] + A[i - 1, j + 1] + A[i, j + 1]) < 0.1:
                            x = 1
                        else:
                            x = self.toss(kr)
                        A[i, j] = A[j, i] = A[Nx1 - i, Nx1 - j] = A[Nx1 - j, Nx1 - i] = x
                        F[i, j] = F[j, i] = F[Nx1 - i, Nx1 - j] = F[Nx1 - j, Nx1 - i] = 0
                        count01 = count01 + 4
                        if A[i, j] > 0.9:
                            count1 = count1 + 4

                if F[i, j] <= 0:
                    if F[i, j + 1] == 1:
                        if (A[i - 1, j] + A[i - 1, j + 1] + A[i, j]) < 0.1:
                            x = 1
                        else:
                            x = self.toss(kr)
                        A[i, j + 1] = A[j + 1, i] = A[Nx1 - i, Nx1 - 1 - j] = A[Nx1 - 1 - j, Nx1 - i] = x
                        F[i, j + 1] = F[j + 1, i] = F[Nx1 - i, Nx1 - 1 - j] = F[Nx1 - 1 - j, Nx1 - i] = 0
                        count01 = count01 + 4
                        if A[i, j + 1] > 0.9:
                            count1 = count1 + 4
        return (A, F, kr)

    def NextStep(self, icg, jcg, ce):
        icgx = icg + self.ND
        jcx = jcg + self.ND
        icgx2 = int(np.floor(icgx / 2))
        jcx2 = int(np.floor(jcx / 2))
        calpha = np.mod(ce, 2)
        cbeta = -1 if ce > 1 else 1
        cgamma = -1 if np.mod(int(icgx + jcx), 4) == 0 else 1
        cg = 1 if self.A[icgx2, jcx2] > 0.5 else 0
        cgd = 1 - cg
        calphad = 1 - calpha
        nalpha = cg * calpha + cgd * calphad
        nbeta = (cg + cgd * cgamma) * cbeta
        nh = (calphad * cgamma * cgd + calpha * cg) * cbeta
        nv = (calpha * cgamma * cgd + calphad * cg) * cbeta
        ing = int(icg + nh * 2)
        jng = int(jcg + nv * 2)
        ingp = icg + cgd * (calphad * cgamma - calpha) * cbeta * 0.5
        jngp = jcg + cgd * (calpha * cgamma - calphad) * cbeta * 0.5
        ne = 0 if nalpha == 0 and nbeta == 1 else (2 if nalpha == 0 else (1 if nbeta == 1 else 3))
        return (ing, jng, ne, ingp, jngp)

    def XNextSteps(self, icgo, jcgo, ceo, Ns):
        ijcx = np.zeros((Ns, 2))
        cex = np.zeros(Ns)
        ijcp = np.zeros((Ns, 2))
        ijcx[0, :] = [icgo, jcgo]
        cex[0] = ceo
        for i in range(Ns - 1):
            ijcx[i + 1, 0], ijcx[i + 1, 1], cex[i + 1], ijcp[i, 0], ijcp[i, 1] = self.NextStep(
                ijcx[i, 0], ijcx[i, 1], cex[i])
        return (ijcx, cex, ijcp)

    def PathCount(self):
        Ns = self.Ns
        ijcx = np.zeros((Ns, 2))
        cex = np.zeros(Ns)
        ijcp = np.zeros((Ns, 2))
        isx = 0
        isa = 0
        ijcx[0, 0] = 2 * np.random.randint(0, 2) - 1
        ijcx[0, 1] = 2 * np.random.randint(0, 2) - 1
        cex[0] = 0
        while isa < Ns - 2:
            isa = isa + 1
            ijcx[isa, 0], ijcx[isa, 1], cex[isa], ijcp[isa - 1, 0], ijcp[isa - 1, 1] = self.NextStep(
                ijcx[isa - 1, 0], ijcx[isa - 1, 1], cex[isa - 1])
            if (int(ijcx[isa, 0]) == int(ijcx[0, 0]) and
                int(ijcx[isa, 1]) == int(ijcx[0, 1]) and
                int(cex[isa]) == int(cex[0])):
                isx = isa
                break
        return isx

    def Dice(self, krRef, Kp, Ki, Nthr):
        Ns = self.Ns
        ijcx = np.zeros((Ns, 2))
        cex = np.zeros(Ns)
        ijcp = np.zeros((Ns, 2))
        krx = np.zeros(Nthr)
        ith = 0
        ithx = 0
        ismax = 0
        while ith < Nthr:
            self.A, self.F, krx[ith] = self.AssignGates(krRef, Kp, Ki)
            Flag1 = Flag2 = isx = isa = 0
            ijcx[0, 0] = ijcx[0, 1] = 1
            cex[0] = 0
            while isa < Ns - 2:
                isa = isa + 1
                ijcx[isa, 0], ijcx[isa, 1], cex[isa], ijcp[isa - 1, 0], ijcp[isa - 1, 1] = self.NextStep(
                    ijcx[isa - 1, 0], ijcx[isa - 1, 1], cex[isa - 1])
                if (int(ijcx[isa, 0]) == int(ijcx[0, 0]) and
                    int(ijcx[isa, 1]) == int(ijcx[0, 1]) and
                    int(cex[isa]) == int(cex[0])):
                    Flag1 = 1
                    isx = isa
                    isa = Ns - 1
            if Flag1 == 1:
                if isx < Ns + 2:
                    Flag2 = 1
                    if isx > ismax:
                        ismax = isx
                        Amax = self.A
                if isx > (Ns / 2) - 2:
                    Flag2 = 2
                    ithx = ith
                    ith = Nthr + 1
            ith = ith + 1
        return (self.A, self.F, Amax, isx, ithx, ismax, Flag1, Flag2, krx)

    def SwitchGate(self, ig, jg):
        Flag = 0
        Ax = self.A * 1
        Fx = self.F * 1
        Nx = self.Nx
        if Ax[ig, jg] < 0.1:
            Ax[ig, jg] = Ax[jg, ig] = Ax[Nx - ig - 1, Nx - jg - 1] = Ax[Nx - jg - 1, Nx - ig - 1] = 1
            Fx[ig, jg] = Fx[jg, ig] = Fx[Nx - ig - 1, Nx - jg - 1] = Fx[Nx - jg - 1, Nx - ig - 1] = 0
            Flag = 1
        elif Ax[ig, jg] > 0.9:
            Ax[ig, jg] = Ax[jg, ig] = Ax[Nx - ig - 1, Nx - jg - 1] = Ax[Nx - jg - 1, Nx - ig - 1] = 0
            Fx[ig, jg] = Fx[jg, ig] = Fx[Nx - ig - 1, Nx - jg - 1] = Fx[Nx - jg - 1, Nx - ig - 1] = 0
            Flag = -1
        return (Ax, Fx, Flag)

    def FlipTestSwitch(self, ksh, iL, iH):
        Ncx = self.PathCount()
        Ns = self.Ns
        Nx2 = int(self.Nx / 2)
        iLx = max(min(iL, Nx2), 1)
        iHx = max(min(iH, Nx2), iLx)
        for ig in range(iLx, iHx):
            for jg in range(ig, self.Nx - 1 - ig):
                Ax = self.A
                Fx = self.F
                if self.F[ig, jg] >= 0:
                    if self.toss(ksh) == 1:
                        self.A, self.F, Flag = self.SwitchGate(ig, jg)
                        Nc = self.PathCount()
                        if Nc < Ncx:
                            self.A = Ax
                            self.F = Fx
                        elif Nc > Ncx:
                            Ncx = Nc
                        if Ncx >= Ns - 5:
                            break
        return Ncx

    def IterFlipTestSwitch(self, ksh, Niter, iL, iH):
        Ns = self.Ns
        Ncx = self.PathCount()
        if Ncx < Ns:
            for iter in range(Niter):
                Ncx = self.FlipTestSwitch(ksh, iL, iH)
                if Ncx >= Ns - 5:
                    break
        return (Ncx, self.A, self.F)

    def PrimitiveCount(self):
        Ax = self.A * 1
        xcacc = np.zeros(4)
        x2acc = np.zeros(2)
        xtot = int((self.ND**2 + 1) / 2)
        for ig in range(self.Nx - 1):
            for jg in range(self.Nx - 1):
                xc = 0
                if (np.mod(ig, 2) == 0 and np.mod(jg, 2) == 0) or (np.mod(ig, 2) == 1 and np.mod(jg, 2) == 1):
                    xc = Ax[ig, jg] + Ax[ig + 1, jg] + Ax[ig + 1, jg + 1] + Ax[ig, jg + 1]
                xc2 = 0
                if xc == 2:
                    xc2 = 2
                    if Ax[ig, jg] + Ax[ig + 1, jg + 1] == 2:
                        xc2 = 1
                    if Ax[ig + 1, jg] + Ax[ig, jg + 1] == 2:
                        xc2 = 1
                if xc > 0:
                    xcacc[int(xc) - 1] = xcacc[int(xc) - 1] + 1
                if xc2 > 0:
                    x2acc[int(xc2) - 1] = x2acc[int(xc2) - 1] + 1
        return (xtot, np.int16([xcacc[0], x2acc[0], x2acc[1], xcacc[2], xcacc[3]]))

    def MapDotMatrix(self, Ncx, ijng, ijngp):
        ND = self.ND
        D = np.zeros((ND, ND))
        for i in range(ND):
            for j in range(ND):
                if np.mod(i + j, 2) != 0:
                    D[i, j] = 100
        for i in range(Ncx):
            if ijngp[i, 0] != ijng[i, 0]:
                D[int(ijngp[i, 0] - 0.5 * (ijng[i, 0] - ND + 1)),
                  int(ijngp[i, 1] - 0.5 * (ijng[i, 1] - ND + 1))] += 2
            if ijngp[i, 0] == ijng[i, 0]:
                if np.mod(ijng[i, 0] + ijng[i, 1], 4) == 0:
                    D[int(0.5 * (ijng[i, 0] + 1 + ND - 1)), int(0.5 * (ijng[i, 1] - 1 + ND - 1))] += 1
                    D[int(0.5 * (ijng[i, 0] - 1 + ND - 1)), int(0.5 * (ijng[i, 1] + 1 + ND - 1))] += 1
                else:
                    D[int(0.5 * (ijng[i, 0] + 1 + ND - 1)), int(0.5 * (ijng[i, 1] + 1 + ND - 1))] += 1
                    D[int(0.5 * (ijng[i, 0] - 1 + ND - 1)), int(0.5 * (ijng[i, 1] - 1 + ND - 1))] += 1
        return D / 2

    def FindBlank(self, D):
        ND = self.ND
        ijb = np.zeros((1, 2))
        ijbd = np.zeros((1, 2))
        Flag = 0
        NBlnk = 0
        for i in range(ND):
            for j in range(ND):
                if D[i, j] < 4:
                    ijb = [(2 * i) - ND + 1, (2 * j) - ND + 1]
                    ijbd = [i, j]
                    Flag = 1
                    NBlnk = NBlnk + 1
        return (ijb, ijbd, Flag, NBlnk)


def generate_kolam(kolam_size, aesthetic_param, kolam_color='#1f77b4', theme='light', complete=False, boundary_type='diamond'):
    """
    Generate a kolam with customizable color, theme, completeness and boundary.
    """
    if kolam_size % 2 == 0:
        kolam_size += 1
    aesthetic_param = max(0.0, min(1.0, aesthetic_param))
    Kp = 0.01
    Ki = 0.0001
    ksh = 0.5
    Niter = 50
    Nthr = 10
    krRef = 1 - aesthetic_param
    if complete:
        KDc = CompleteKolamDraw(kolam_size)
        A2, F2, A2max, isx, ithx, ismax, Flag1, Flag2, krx2 = KDc.Dice(krRef, Kp, Ki, Nthr)
        _ = KDc.PathCount()
        Nx2x = int((kolam_size + 1) / 2)
        Ncx, GM, GF = KDc.IterFlipTestSwitch(ksh, Niter, 1, Nx2x)
        Ns = 2 * (kolam_size**2 + 1) + 5
        ijng, ne, ijngp = KDc.XNextSteps(1, 1, 1, Ns)
        # Optionally evaluate completeness; we focus on plotting for UI
        fig = plotkolam(ijngp, kolam_color, theme)
        return fig
    else:
        KD = KolamDraw(kolam_size)
        KD.set_boundary(boundary_type)
        A2, F2, A2max, isx, ithx, ismax, Flag1, Flag2, krx2 = KD.Dice(krRef, Kp, Ki, Nthr)
        _ = KD.PathCount()
        Nx2x = int((kolam_size + 1) / 2)
        Ncx, GM, GF = KD.IterFlipTestSwitch(ksh, Niter, 1, Nx2x)
        Ns = 2 * (kolam_size**2 + 1) + 5
        icgo = 1
        jcgo = 1
        ceo = 1
        ijng, ne, ijngp = KD.XNextSteps(icgo, jcgo, ceo, Ns)
        fig = plotkolam(ijngp, kolam_color, theme)
        return fig