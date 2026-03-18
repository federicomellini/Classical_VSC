import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt 
from matplotlib import cm
from scipy.signal import find_peaks
from heapq import nlargest
import os 





'''Define accelerations based on different hamiltonians'''


import numpy as np



def Pauli_Fierz(xc, vc, xm_values, vm_values, num_mol, param_cav, param_mol, t):
    wc, E = param_cav
    wm = np.array(param_mol[0:num_mol])
    lam = param_mol[-1]
    
    sum_xm = np.sum(xm_values)
    a_xc = -(wc**2) * xc + lam * wc * sum_xm
    a_xm_values = -xm_values * wm**2 + lam * xc * wc - (lam**2) * sum_xm
    
    return a_xc, a_xm_values

def Pauli_Fierz_driven(xc, vc, xm_values, vm_values, num_mol, param_cav, param_mol, t):
    wc, E = param_cav
    wm = np.array(param_mol[0:num_mol])
    lam = param_mol[-1]
    
    sum_xm = np.sum(xm_values)
    a_xc = E * np.cos(wc * t) - (wc**2) * xc + lam * wc * sum_xm
    a_xm_values = -xm_values * wm**2 + lam * xc * wc - (lam**2) * sum_xm
    
    return a_xc, a_xm_values

def Pauli_Fierz_static_global(xc, vc, xm_values, vm_values, num_mol, param_cav, param_mol, t):
    wc, E = param_cav
    wm = np.array(param_mol[0:num_mol])
    lam = param_mol[-1]
    
    sum_xm = np.sum(xm_values)
    a_xc = -(wc**2) * xc + lam * wc * sum_xm 
    a_xm_values = E - xm_values * wm**2 + lam * xc * wc - (lam**2) * sum_xm
    
    return a_xc, a_xm_values

def Pauli_Fierz_static_local(xc, vc, xm_values, vm_values, num_mol, param_cav, param_mol, t):
    wc, E = param_cav
    wm = np.array(param_mol[0:num_mol])
    lam = param_mol[-1]
   
    sum_xm = np.sum(xm_values)
    a_xc = -xc * wc**2 + lam *  wc *sum_xm 
    a_xm_values = -xm_values * wm**2 + lam * xc * wc - (lam**2) * sum_xm
    
    # Apply Stark shift only to the 0-th molecule
    a_xm_values[0] += E
    
    return a_xc, a_xm_values

#------------------------------------------------------------------------------------------#

''' Define Velocity Verlet algorithm'''
def velocity_verlet_old(acceleration, init_cond, time_points, num_mol, param_cav, param_mol):
    """
    Velocity verlet algorith implementing the generalized langevin EOM
    from Tuckerman, eq (15.5.17)
    """
    n_points = len(time_points)
    dt = time_points[1] - time_points[0]
    
    # Initialize arrays to store positions and velocities
    xc_values = np.zeros(n_points)
    vc_values = np.zeros(n_points)
    xm_values = np.zeros((num_mol, n_points))
    vm_values = np.zeros((num_mol, n_points))
    
    # Set initial conditions
    xc_values[0], vc_values[0] = init_cond[:2]
    xm_values[:, 0], vm_values[:, 0] = init_cond[2:num_mol+2], init_cond[num_mol+2:(num_mol+2)+num_mol]
    
    for i in range(n_points - 1):
        t = time_points[i]
        
        # Calculate current accelerations
        a_xc, a_xm_values = acceleration(xc_values[i], vc_values[i], xm_values[:, i], vm_values[:, i], num_mol, param_cav, param_mol, t)
        
        # Update positions
        R_c = sigma_c * dt**(3 / 2) * (0.5*np.random.normal(0.0, 1.0, 1) + 1/(2*np.sqrt(3))*np.random.normal(0.0, 1.0, 1)) # xi and theta
        #print('current xm_values[:,i]:', xm_values[:,i])
        xc_values[i + 1] = xc_values[i] + vc_values[i] * dt + 0.5 * (a_xc - k * vc_values[i]) * dt**2  + R_c.item()

        for j in range(num_mol):
            R_m = sigma_m * dt ** (3 / 2) * (0.5*np.random.normal(0.0, 1.0, 1) + 1/(2*np.sqrt(3))*np.random.normal(0.0, 1.0, 1))  # xi and theta
            #print('current R_m:', R_m)
            xm_values[j, i + 1] = xm_values[j, i] + vm_values[j, i] * dt + 0.5 * (a_xm_values[j] - lamb * vm_values[j, i]) * dt ** 2 + R_m.item()
        
        # Calculate new accelerations at the new positions
        a_xc_new, a_xm_values_new = acceleration(xc_values[i + 1], vc_values[i], xm_values[:, i + 1], vm_values[:, i], num_mol, param_cav, param_mol, t + dt)
        
        # Update velocities
        vc_values[i + 1] = vc_values[i] + 0.5 * (a_xc + a_xc_new) * dt - k * vc_values[i]*dt + sigma_c * np.sqrt(dt) * np.random.normal(0.0, 1.0, 1).item() - k *(0.5 * dt ** 2 * (a_xc - k * vc_values[i]) + R_c.item())
        for j in range(num_mol):
            vm_values[j, i + 1] = vm_values[j, i] + 0.5 * (a_xm_values[j] + a_xm_values_new[j]) * dt - lamb * vm_values[j, i]*dt + sigma_m * np.sqrt(dt) * np.random.normal(0.0, 1.0, 1).item() - lamb * (0.5 * dt ** 2 * (a_xm_values[j] - lamb * vm_values[j, i]) + R_m.item())
    
    return xc_values, vc_values, xm_values, vm_values


def velocity_verlet(acceleration, init_cond, time_points, num_mol, param_cav, param_mol):
    """
    Velocity verlet algorithm implementing the generalized langevin EOM
    from Tuckerman, eq (15.5.17)
    """
    n_points = len(time_points)
    dt = time_points[1] - time_points[0]
    
    # Pre-calculate time step constants to save CPU cycles inside the loop
    dt_2 = dt**2
    dt_12 = np.sqrt(dt)
    dt_32 = dt**(3 / 2)
    sqrt3_inv = 1 / (2 * np.sqrt(3))
    
    # Initialize arrays to store positions and velocities
    xc_values = np.zeros(n_points)
    vc_values = np.zeros(n_points)
    xm_values = np.zeros((num_mol, n_points))
    vm_values = np.zeros((num_mol, n_points))
    
    # Set initial conditions
    xc_values[0], vc_values[0] = init_cond[:2]
    xm_values[:, 0] = init_cond[2:num_mol+2]
    vm_values[:, 0] = init_cond[num_mol+2:(num_mol+2)+num_mol]
    
    # Calculate initial acceleration ONCE before the loop begins
    a_xc, a_xm = acceleration(xc_values[0], vc_values[0], xm_values[:, 0], vm_values[:, 0], num_mol, param_cav, param_mol, time_points[0])
    
    for i in range(n_points - 1):
        t = time_points[i]
        
        # Vectorized noise generation: generate random arrays for all molecules at once
        xi_c, theta_c = np.random.normal(0.0, 1.0, 2)
        xi_m = np.random.normal(0.0, 1.0, num_mol)
        theta_m = np.random.normal(0.0, 1.0, num_mol)
        
        # Calculate stochastic position updates
        R_c = sigma_c * dt_32 * (0.5 * xi_c + sqrt3_inv * theta_c)
        R_m = sigma_m * dt_32 * (0.5 * xi_m + sqrt3_inv * theta_m)
        
        # Update positions
        xc_values[i + 1] = xc_values[i] + vc_values[i] * dt + 0.5 * (a_xc - k * vc_values[i]) * dt_2 + R_c
        xm_values[:, i + 1] = xm_values[:, i] + vm_values[:, i] * dt + 0.5 * (a_xm - lamb * vm_values[:, i]) * dt_2 + R_m
        
        # Calculate NEW accelerations at the new positions
        a_xc_new, a_xm_new = acceleration(xc_values[i + 1], vc_values[i], xm_values[:, i + 1], vm_values[:, i], num_mol, param_cav, param_mol, t + dt)
        
        # Update velocities
        vc_values[i + 1] = (vc_values[i] + 0.5 * (a_xc + a_xc_new) * dt - k * vc_values[i] * dt 
                            + sigma_c * dt_12 * xi_c - k * (0.5 * dt_2 * (a_xc - k * vc_values[i]) + R_c))
        
        vm_values[:, i + 1] = (vm_values[:, i] + 0.5 * (a_xm + a_xm_new) * dt - lamb * vm_values[:, i] * dt 
                               + sigma_m * dt_12 * xi_m - lamb * (0.5 * dt_2 * (a_xm - lamb * vm_values[:, i]) + R_m))
        
        # Shift new accelerations to current for the next time step
        a_xc, a_xm = a_xc_new, a_xm_new
    
    return xc_values, vc_values, xm_values, vm_values



def check_temperature_consistency(vc_values, vm_values, kT, num_mol):
    """
    This function checks if the time-averaged velocities (photon + molecules) are consistent with the initial temperature.
    
    Parameters:
    - vc_values: Array of photon velocities (1D array).
    - vm_values: Array of molecular velocities (2D array, each row corresponds to a molecule).
    - kT: The value of k_B * T (in atomic units).
    - num_mol: Number of molecules.
    
    Returns:
    - Consistency check for the system (photon + molecules).
    """
    
    # Calculate the time-averaged <v^2> for the photon
    v2_photon = np.mean(vc_values[-1000:] ** 2)

    # Calculate the time-averaged <v^2> for each molecule
    v2_molecules = np.zeros(num_mol)

    for j in range(num_mol):
        v2_molecules[j] = np.mean(vm_values[j, -400:] ** 2)  # Average over time for each molecule
    
    # Average <v^2> across all molecules
    avg_v2_molecules = np.mean(v2_molecules)



    # Combine the photon and molecule contributions
    total_avg_v2 = (v2_photon + avg_v2_molecules * num_mol) / (num_mol + 1)
    
    #print("Time-averaged <v^2> for photon:", v2_photon)
    #print("Average <v^2> for molecules:", avg_v2_molecules)
    #print("Combined average <v^2> for system (photon + molecules):", total_avg_v2)
    #print("Expected <v^2> from temperature (kT):", kT)
    
    # Compare with the expected value from temperature (kT)
    system_consistent = np.isclose(total_avg_v2, kT, rtol=0.8)

    if system_consistent:
        message =("The time-averaged velocities of the photon and molecules are consistent with the initial temperature.")
    else:
        message =("The time-averaged velocities of the photon and molecules are NOT consistent with the initial temperature.")

    return message



''' Define analysis tool functions '''

def autocorr(values):
    '''
    Computes autocorrelation of either positional
    values (C_xx) or velocity values (C_vv). 
    '''
    results = np.correlate(values, values, mode='full')
    results = results[results.size // 2:]
    autocorr = results / results[0]
    return autocorr



def crosscorr(values_1, values_2):
    results = np.correlate(values_1, values_2, mode='full')
    results = results[results.size // 2:]
    norm_1 = np.sum(values_1**2)
    norm_2 = np.sum(values_2**2)
    C_12 = results / np.sqrt(norm_1 * norm_2)
    return C_12


def fft_autocorr(autocorr):
    '''
    Implement zero-padding (e.g., 9 times the original length)
    This increases the FT bin density by a factor of 10 and 
    makes the FT plots nicer and smoother.
    '''  
    pad_length = len(autocorr) * 9
    padded_autocorr = np.pad(autocorr, (0, pad_length), mode='constant')
    
    fft = np.fft.fft(padded_autocorr)
    fftfreq = 2*np.pi*np.fft.fftfreq(fft.shape[-1], d=t_step)
    
    return fft, fftfreq


def bright_autocorr(xm_values):
    """
    Calculates the autocorrelation of the macroscopic bright mode.
    xm_values should be a 2D array of shape (num_mol, n_points).
    """
    # Sum along the molecule axis (axis=0) to get total polarization over time
    P_t = np.sum(xm_values, axis=0)
    
    results = np.correlate(P_t, P_t, mode='full')
    results = results[results.size // 2:]
    C_PP = results / results[0]
    return C_PP

def calc_energies(xm_values, vm_values, freqs, mass=1.0):
    """
    Helper function to calculate the total energy of each molecule over time.
    Requires 2D position/velocity arrays and a 1D array of molecular frequencies.
    """
    # Reshape frequencies so they broadcast correctly across the time points
    w2 = (freqs**2)[:, np.newaxis]
    
    # Calculate E = 1/2*m*v^2 + 1/2*m*w^2*x^2 for all molecules and all times
    energies = 0.5 * mass * vm_values**2 + 0.5 * mass * w2 * xm_values**2
    return energies

def calc_ipr(energies):
    """
    Calculates the Inverse Participation Ratio (IPR) over time.
    energies should be a 2D array of shape (num_mol, n_points).
    """
    # Sum of squared energies at each time step
    sum_sq = np.sum(energies**2, axis=0)
    
    # Square of the summed energies at each time step
    sq_sum = (np.sum(energies, axis=0))**2
    
    # Calculate IPR, avoiding division by zero if energy is perfectly 0
    ipr_t = np.divide(sum_sq, sq_sum, out=np.zeros_like(sum_sq), where=sq_sum!=0)
    return ipr_t



#-------------------------------------------------------------------------------------#

if __name__ == "__main__":

    # Definite conversion factors:

    aut_to_fs = 0.0241888
    au_to_ev  = 27.2114


    ##########################################
    ''' DEFINE PARAMETERS (Thermalization) '''
    ##########################################

    # Set up number of molecules:
    num_mol = 15
    
    # Set up parameters
    wc = 0.005512
    wm = 0.005512 # freqs au of a C=C bond


    E0 = 0.0001 # Amplitude of driving laser
    param_cav = [wc, E0]  # wc, E
    freqs = np.random.normal(wm, 0.0, num_mol)  # wm
    gc =0.0*wm 
    lam = gc/np.sqrt(num_mol) # light-matter coupling
    param_mol = freqs.tolist() + [lam] # wm, lamba

    # Set up friction coefficients k, lamb and random kick sigma: 
    k    = 1.0e-5 # Friction coeff for photon
    lamb = 1.0e-5 # Friction coeff for molecules
    kT   = 1.0*9.44e-4 # 9.44x10⁻4 au is value of room temperature energy 25,7 meV
    beta = 1/kT
    mu   = 1
    sigma_c = np.sqrt(2*kT*k/mu)
    sigma_m = np.sqrt(2*kT*lamb/mu)

    # Not really that important how the initial conditions are chosen, because of Markovianity
    # Molecules
    std_x = 1/np.sqrt(beta)
    std_v = 1/np.sqrt(beta * wm**2)*0.05

    init_xm = np.random.normal(0, std_x, num_mol)
    init_vm = np.random.normal(0, std_v, num_mol)

    # Photon
    I = 3*wc/2 # energy when cavity excited with one photon
    theta = np.random.uniform(0, 2*np.pi,1).item()

    init_xc = np.sqrt(2 * I / (wc)) * np.sin(theta)
    init_vc = np.sqrt(2 * I * wc) * np.cos(theta) * 0.05



    # Combine initial conditions into a single list
    init_cond = [init_xc, init_vc] + init_xm.tolist() + init_vm.tolist()


    # Define time pointsp
    t_eq = 350000
    t_step  = 10
    time_points = np.arange(0, t_eq, t_step)  # Time points from 0 to 10
  

    # Run the equilibration consistency check a few times to ensure equilibration is reached6 

    for i in range(10):
        xc_values_eq, vc_values_eq, xm_values_eq, vm_values_eq = velocity_verlet(Pauli_Fierz, init_cond, time_points, num_mol, param_cav, param_mol)
        message = check_temperature_consistency(vc_values_eq, vm_values_eq, kT, num_mol)

        if message == "The time-averaged velocities of the photon and molecules are consistent with the initial temperature.":
            print(message)
            break
    else:
        print('\n ' \
        'EQUIBILBRATION FAILED! \n' \
        'modify system parameters, script terminated')
        #exit()
            


    
    ''' Take initial condition from equilibration run and propagate '''


    # Molecules
    init_xm = xm_values_eq[:,-1]
    init_vm = vm_values_eq[:,-1]

    # Photon 
    init_xc = xc_values_eq[-1]
    init_vc = vc_values_eq[-1]

    init_cond = [init_xc, init_vc] + init_xm.tolist() + init_vm.tolist()

    # Redefine new parameters for the propagation

    #E0 = 0.0 # Amplitude of driving laser
    param_cav = [wc, E0]  # wc, E
    gc =0.1*wc
    lam = gc/np.sqrt(num_mol) # light-matter coupling
    param_mol = freqs.tolist() + [lam] # wm, lamba

    # Define time points
    t_final = 1000000
    time_points = np.arange(0, t_final, t_step)  # Time points from 0 to 10

    # Solve the system using Velocity-Verlet algorithm
    xc_values, vc_values, xm_values, vm_values = velocity_verlet(Pauli_Fierz_static_local, init_cond, time_points, num_mol, param_cav, param_mol)


    ''' Calculate things '''
    

    C_xcxc = autocorr(xm_values[0])
    C_vcvc = autocorr(vm_values[0])
    C_xx_bright = bright_autocorr(xm_values)
    C_vv_bright = bright_autocorr(vm_values)

    fft_xcxc, fftfreq_xcxc = fft_autocorr(C_xcxc)
    fft_vcvc, fftfreq_vcvc = fft_autocorr(C_vcvc)
    fft_xx_bright, fftfreq_xx_bright = fft_autocorr(C_xx_bright)
    fft_vv_bright, fftfreq_vv_bright = fft_autocorr(C_vv_bright)
    

    energies = calc_energies(xm_values, vm_values, freqs)
    ipr = calc_ipr(energies)

    # Average position of all molecules
    av_pos = np.mean(xm_values)
    print('Average position of the molecules:', av_pos)


    ''' PLOT '''

    fig, axs = plt.subplots(2, 2, figsize=(8,4), constrained_layout=True, sharex='col')

    #xmin = 0
    #xmax = 100

    # Axis thickness and label font size
    axis_thickness = 1.5
    label_fontsize = 12
    title_fontsize = 14

    # Set thicker axis lines globally
    for ax in axs.flat:
        #ax.set_ylim(0,0.003)
        ax.tick_params(width=axis_thickness, labelsize=label_fontsize)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(1,-1))
        ax.spines['top'].set_linewidth(axis_thickness)
        ax.spines['right'].set_linewidth(axis_thickness)
        ax.spines['bottom'].set_linewidth(axis_thickness)
        ax.spines['left'].set_linewidth(axis_thickness)

    #--------------------------
    ''' subplot1 L '''
    ax00 = axs[0,0]
    ax00.plot(time_points*aut_to_fs, C_xcxc, label='C_xcxc(t)', color='darkorange')
    ax00.legend(loc="upper right")
    ax00.set_xlabel('Time (fs)')
    ax00.set_ylabel('Photon position autocorrelation')

    ax01 = axs[0,1]
    ax01.plot(time_points*aut_to_fs, C_vcvc, label='C_vcvc(t)', color='darkorange')
    ax01.legend(loc="upper right")
    ax01.set_xlabel('Time (fs)')
    ax01.set_ylabel('Photon velocity autocorrelation')

    ax10 = axs[1,0]
    ax10.plot(time_points*aut_to_fs, C_xx_bright, label='C_xx_bright(t)', color='tab:blue')
    ax10.legend(loc="upper right")
    ax10.set_xlabel('Time (fs)')
    ax10.set_ylabel('Bright state position autocorrelation')

    ax11 = axs[1,1]
    ax11.plot(time_points*aut_to_fs, C_vv_bright, label='C_vv_bright(t)', color='tab:blue')
    ax11.legend(loc="upper right")
    ax11.set_xlabel('Time (fs)')
    ax11.set_ylabel('Bright state velocity autocorrelation')

    plt.show()




    fig, axs = plt.subplots(2, 2, figsize=(8,4), constrained_layout=True, sharex='col')

    #xmin = 0
    #xmax = 100

    # Axis thickness and label font size
    axis_thickness = 1.5
    label_fontsize = 12
    title_fontsize = 14

    # Set thicker axis lines globally
    for ax in axs.flat:
        ax.set_xlim((wc-2.5*lam)*au_to_ev , (wc+2.5*lam)*au_to_ev)
        ax.set_ylim(0.0,1.1)
        ax.tick_params(width=axis_thickness, labelsize=label_fontsize)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(1,-1))
        ax.spines['top'].set_linewidth(axis_thickness)
        ax.spines['right'].set_linewidth(axis_thickness)
        ax.spines['bottom'].set_linewidth(axis_thickness)
        ax.spines['left'].set_linewidth(axis_thickness)

    #--------------------------
    ''' subplot1 L '''
    ax00 = axs[0,0]
    ax00.plot(fftfreq_xcxc*au_to_ev, np.abs(fft_xcxc.real)/np.max(np.abs(fft_xcxc.real)), label='FT(C_xcxc(t))', color='darkorange')
    ax00.legend(loc="upper right")
    ax00.set_xlabel('Frequency (eV)')
    ax00.set_ylabel('Photon position FT')

    ax01 = axs[0,1]
    ax01.plot(fftfreq_vcvc*au_to_ev, np.abs(fft_vcvc.real)/np.max(np.abs(fft_vcvc.real)), label='FT(C_vcvc(t))', color='darkorange')
    ax01.legend(loc="upper right")
    ax01.set_xlabel('Frequency (eV)')
    ax01.set_ylabel('Photon velocity FT')

    ax10 = axs[1,0]
    ax10.plot(fftfreq_xx_bright*au_to_ev, np.abs(fft_xx_bright.real)/np.max(np.abs(fft_xx_bright.real)), label='FT(C_xx_bright(t))', color='tab:blue')
    ax10.legend(loc="upper right")
    ax10.set_xlabel('Frequency (eV)')
    ax10.set_ylabel('Bright state position autocorrelation')

    ax11 = axs[1,1]
    ax11.plot(fftfreq_vv_bright*au_to_ev, np.abs(fft_vv_bright.real)/np.max(np.abs(fft_vv_bright.real)), label='FT(C_vv_bright(t))', color='tab:blue')
    ax11.legend(loc="upper right")
    ax11.set_xlabel('Frequency (eV)')
    ax11.set_ylabel('Bright state velocity autocorrelation')

    plt.show()



   



    # Plot the results
    cmap = cm.get_cmap("coolwarm")
    cols = cmap(np.linspace(0,1,num_mol))

    plt.figure(figsize=[12,6])
    plt.title('Coupled Generalized Langevin')
    plt.plot(time_points[-500:]*aut_to_fs, xc_values[-500:], label='xc(t)', color='black')
    plt.plot(time_points[-500:]*aut_to_fs, xm_values[i,-500:], label=f'xm0(t)', color='tab:red')
    for i in range(1, num_mol):
        plt.plot(time_points[-500:]*aut_to_fs, xm_values[i,-500:], label=f'xm{i+1}(t)', color='tab:cyan', alpha=0.4)
    #plt.ylim(-4,8)
    plt.xlabel('Time (fs)')
    plt.ylabel('Values')

    #plt.legend()

    plt.show()


    plt.figure(figsize=[12,6])
    plt.title('IPR')
    #for i in range(num_mol):
    plt.plot(time_points*aut_to_fs, ipr, label='ipr', color='tab:blue')
    plt.xlabel('Time (fs)')
    plt.ylabel('Values')
    plt.show()
    #plt.legend()


    exit()

    # Plot the results
    plt.figure(figsize=[6,12])
    plt.title('Autocorrelation function')
    plt.plot(time_points*aut_to_fs, C_xcxc, label='C_xcxc(t)', color='black')
    #for i in range(num_mol):
    #    plt.plot(time_points*aut_to_fs, xm_values_eq[i], label=f'xm{i+1}(t)')
    #plt.ylim(-4,8)
    plt.xlabel('Time (fs)')
    plt.ylabel('Values')

    #plt.legend()

    plt.show()


    plt.figure(figsize=[12,6])
    plt.title('Autocorrelation function')
    plt.plot(fftfreq, np.abs(fft.real)/np.max(np.abs(fft.real)), color='black', label='xc_values')
    plt.xlim(0.0, 0.01)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Time (fs)')
    plt.ylabel('Values')

    #plt.legend()

    plt.show()




    # Save initial positions to a text file
    with open("initial_positions.txt", "w") as f_pos:
        # Save init_xc and init_xm in separate lines
        f_pos.write(f"{init_xc}\n")
        f_pos.write(' '.join(map(str, init_xm)))

    # Save initial velocities to a text file
    with open("initial_velocities.txt", "w") as f_vel:
        # Save init_vc and init_vm in separate lines
        f_vel.write(f"{init_vc}\n")
        f_vel.write(' '.join(map(str, init_vm)))


    ''' Define path of directory for saving stuff'''
    E_ampl =0
    dir = f"{gc}_{E_ampl}"


    # Create the directory if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    thermalization_path = os.path.join(dir, 'thermalization.txt')

    with open(thermalization_path, 'w') as readme_file:
        readme_file.write(message)





    
   
