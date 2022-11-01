'''
    This is a python script to compute Hayashi spectra follwing Randel & Held (1991).
    These are spectra of power spectral density in phase speed-wavenumber space in <units>**2 * s / m.
    Specifically this code is desiged to work for xarray.DataArrays
    Based on ncl code by Bernat Jimenez Esteve.
    
    Wolfgang Wicker, wolfgang.wicker(at)env.ethz.ch, April 2021
    
    Modified by Bernat Jimenez, June 2021    
'''
import numpy as np
import xarray as xr
from numba import float64, complex128, guvectorize



# Bernat uses the 'mjo_wavenum_freq_season' NCL routine

@guvectorize(
    "(float64[:,:],float64[:],float64[:],complex128[:,:])",
    "(m,n), (k), (l) -> (k,l)",
    #nopython=True
    forceobj=True
)
def fft2d(a,freq0,freq1,out):
    '''
        vectorized two-dimensional fast fourier transfrom
        
        - take care of the normalization
        - currently doesn't work for dask arrays
    '''
    out[:,:] = np.fft.fft2(a,norm=None)
    
    
    
def wavenum_freq_spect(da,timestep=86400):
    '''
        wavenumber-frequency power spectrum
        
        - takes xarray.DataArray input
        - sampling is sample spacing in seconds
        - tapering produces a smoother spectrum but increases leakage
    '''
    print('\n Time-mean, zonal-mean variance')
    print(da.var(('time','longitude')))

    # prepare arrays for frequency, wavenumber, tapering
    freq = xr.DataArray(np.fft.fftfreq(len(da.time),d=timestep),dims=('frequency'))
    wavenum = xr.DataArray(np.fft.fftfreq(len(da.longitude),d=1/len(da.longitude)),dims=('wavenumber'))
    taper = xr.DataArray(np.hanning(len(da.time)),coords=dict(time=da.time),dims=('time'))
    
    # apply tapering
    da = da * taper
    
    # fft2d currently doesn't work for dask arrays
    da = da.compute() 
    spect = xr.apply_ufunc(fft2d,
                           *(da,freq,wavenum),
                           input_core_dims=[['time','longitude'],['frequency'],['wavenumber']],
                           output_core_dims=[['frequency','wavenumber']],
                           dask='parallelized',
                           output_dtypes=[np.complex128])
    spect = spect.assign_coords(dict(frequency=freq,wavenumber=wavenum))
    
    # Compute power spectrum
    spect = spect * np.conjugate(spect)
    spect = np.real(spect)
    
    # this is in units of power spectral denstity times frequency resolution times wavenumber resolution
    #spect = spect / len(da.time)**2 / len(da.longitude)**2
    # frequency resolution is one over timeseries length
    spect = spect / len(da.longitude)**2 / len(da.time) * timestep
    # use positive wavenumber only
    spect = spect.where(wavenum >= 0,drop=True) * 2
    
    # account for tapering
    spect = spect / (taper**2).mean()
    
    print('\n Variance retainded by integration over frequency and wavenumber')
    print(spect.integrate(('frequency','wavenumber')))
    
    return spect



@guvectorize(
    "(float64[:],float64[:],float64[:],float64[:])",
    "(n), (n), (m) -> (m)",
    nopython=True
)
def interp_freq(spect,freq,fc,out):
    '''
    '''
    out[:] = np.interp(freq,fc,spect)

    

def freq2phase_speed(spect,dc=1,cmax=30):
    '''
        Calculate wavenumber-phase speed spectra from wavenumber-frequency spectra
        following Randel & Held (1991)
        
        -positive phase speed is eastward
    '''
    # Define an array of phase speed
    c = xr.DataArray(np.arange(-1*cmax,cmax+dc,dc),dims=('phase_speed')) 
    c = c.assign_coords(phase_speed=c)
    
    # Define the array of frequencies that correspond that phase speeds
    a = 6371000
    factor = 1 / (2*np.pi*a)
    factor = factor / np.cos(spect.latitude/180*np.pi)
    fc = factor * c * spect.wavenumber # this has dimenstions latitude, phase speed, wavenumber
    
    # Interpolate linearly to these frequencies
    spect = spect.sortby('frequency')
    
    new_spect = xr.DataArray(coords=[spect.latitude, c, spect.wavenumber],
                             dims=('latitude','phase_speed','wavenumber')) 
    
    for i,lat in enumerate(spect.latitude):
        for k,wn in enumerate(spect.wavenumber):
            new_spect[i,:,k] = spect[i,:,k].interp(frequency=fc[i,:,k],method="nearest")
            
    #new_spect = xr.apply_ufunc(interp_freq,
    #                           *(spect,spect.frequency,fc),
    #                           input_core_dims=[['frequency'],['frequency'],['phase_speed']],
    #                           output_core_dims=[['phase_speed']],
    #                           dask='parallelized',
    #                           output_dtypes=[spect.dtype])
    
    # scale power spectral density into units of phase speed
    new_spect = new_spect * spect.wavenumber * factor
    # positive phase speed is eastward
    new_spect['phase_speed'] = -1* new_spect['phase_speed']
    
    print('\n Variance retainded by integration over phase speed and wavenumber')
    print((-1)*new_spect.integrate(('phase_speed','wavenumber')))
    
    return new_spect
    
    