from __future__ import print_function

import numpy as np
import healpy as hp
try :
    from lenspyx.shts import shts
    from lenspyx.bicubic import bicubic
except:
    print('Could not load fortran modules')
    pass
from lenspyx import utils
from lenspyx import angles

def alm2lenmap(alm, dlms, nside, facres=0, nband=8, verbose=True):
    r"""Computes a deflected spin-0 healpix map from its alm and deflection field alm.

        Args:
            alm: undeflected map healpy alm array
            dlms: The spin-1 deflection, in the form of either a list of two healpy alm arrays or a list of two healpix maps.

                    In the former case the two arrays are the gradient and curl deflection healpy alms.
                    (e.g. :math:`\sqrt{L(L+1)}\phi_{LM}` with :math:`\phi` the lensing potential)
                    The curl can be set to None if irrelevant.

                    In the latter case the two arrays are the real and imag. part of the precomputed spin-1 deflection.
                    (e.g. :math:`-\sum_{LM}\sqrt{L(L+1)}\phi_{LM} \:_{1}Y_{LM}(\hat n))`
                    This saves some computation time as this operation must be performed anyways.
                    In this case, their healpix resolution must match the argument *nside*.

            nside: desired healpix resolution of the deflected map
            facres(optional): the deflected map is constructed by interpolation of the undeflected map,
                              built at target res. ~ :math:`0.7 * 2^{\rm facres}.` arcmin
            nband(optional): To avoid dealing with too many large maps in memory, the operations is split in bands.
            verbose(optional): If set, prints a bunch of timing and other info. Defaults to true.

        Returns:
            Deflected healpy map at resolution nside.


    """
    assert len(dlms) == 2
    if not np.iscomplexobj(dlms[0]): assert dlms[0].size == dlms[1].size and dlms[0].size == hp.nside2npix(nside)
    return _lens_gclm_sym_timed(0, dlms[0], -alm, nside, dclm=dlms[1], nband=nband, facres=facres, verbose=verbose)

def alm2lenmap_spin(gclm, dlms, nside, spin, nband=8, facres=-1, verbose=True):
    r"""Computes a deflected spin-weight Healpix map from its gradient and curl modes and deflection field alm.

        Args:
            gclm: list with undeflected map healpy gradient and curl array (e.g. polarization Elm and Blm).
                    The curl mode can be set to None if irrelevant.

            dlms: The spin-1 deflection, in the form of either a list of two healpy alm arrays or a list of two healpix maps.

                    In the former case the two arrays are the gradient and curl deflection healpy alms.
                    (e.g. :math:`\sqrt{L(L+1)}\phi_{LM}` with :math:`\phi` the lensing potential)
                    The curl can be set to None if irrelevant.

                    In the latter case the two arrays are the real and imag. part of the precomputed spin-1 deflection.
                    (e.g. :math:`-\sum_{LM}\sqrt{L(L+1)}\phi_{LM} \:_{1}Y_{LM}(\hat n))`
                    This saves some computation time as this operation must be performed anyways.
                    In this case, their healpix resolution must match the argument *nside*.

            nside: desired healpix resolution of the deflected map
            spin: spin-weight of the maps to deflect (2 for polarization).
            facres(optional): the deflected map is constructed by interpolation of the undeflected map,
                              built at target res. ~ :math:`0.7 * 2^{\rm facres}.` arcmin.
            nband(optional): To avoid dealing with too many large maps in memory, the operations is split in bands.
            verbose(optional): If set, prints a bunch of timing and other info. Defaults to true.

        Returns:
            Deflected healpy maps at resolution nside (real and imaginary parts).


    """
    assert len(gclm) == 2
    assert len(dlms) == 2
    if not np.iscomplexobj(dlms[0]): assert dlms[0].size == dlms[1].size and dlms[0].size == hp.nside2npix(nside)
    ret = _lens_gclm_sym_timed(spin, dlms[0], gclm[0], nside,
                                clm=gclm[1], dclm=dlms[1], nband=nband, facres=facres, verbose=verbose)
    return ret.real, ret.imag

### Modified by FG
def strip_alm2lenmap_spin(gclm, dlms, nside, spin, theta1, theta2, nband=4, facres=-1, verbose=True):
    r"""Computes a deflected spin-weight Healpix map from its gradient and curl modes and deflection field alm.

        Args:
            gclm: list with undeflected map healpy gradient and curl array (e.g. polarization Elm and Blm).
                    The curl mode can be set to None if irrelevant.

            dlms: The spin-1 deflection, in the form of either a list of two healpy alm arrays or a list of two healpix maps.

                    In the former case the two arrays are the gradient and curl deflection healpy alms.
                    (e.g. :math:`\sqrt{L(L+1)}\phi_{LM}` with :math:`\phi` the lensing potential)
                    The curl can be set to None if irrelevant.

                    In the latter case the two arrays are the real and imag. part of the precomputed spin-1 deflection.
                    (e.g. :math:`-\sum_{LM}\sqrt{L(L+1)}\phi_{LM} \:_{1}Y_{LM}(\hat n))`
                    This saves some computation time as this operation must be performed anyways.
                    In this case, their healpix resolution must match the argument *nside*.

            nside: desired healpix resolution of the deflected map
            spin: spin-weight of the maps to deflect (2 for polarization).
            facres(optional): the deflected map is constructed by interpolation of the undeflected map,
                              built at target res. ~ :math:`0.7 * 2^{\rm facres}.` arcmin.
            nband(optional): To avoid dealing with too many large maps in memory, the operations is split in bands.
            verbose(optional): If set, prints a bunch of timing and other info. Defaults to true.

        Returns:
            Deflected healpy maps at resolution nside (real and imaginary parts).


    """
    assert len(gclm) == 2
    assert len(dlms) == 2
    assert 0<=theta1<=np.pi, 'theta should be in [0,pi], got %.3f'%theta1
    assert 0<=theta2<=np.pi, 'theta should be in [0,pi], got %.3f'%theta2
    if theta1>=theta2:
        theta1, theta2 = theta2, theta1
    if not np.iscomplexobj(dlms[0]): assert dlms[0].size == dlms[1].size and dlms[0].size == hp.nside2npix(nside)
    ret = _strip_lens_gclm_sym_timed(spin, dlms[0], gclm[0], nside, theta1, theta2,
                                clm=gclm[1], dclm=dlms[1], nband=nband, facres=facres, verbose=verbose)
    return ret.real, ret.imag

def _lens_gclm_sym_timed(spin, dlm, glm, nside, nband=8, facres=0, clm=None, dclm=None, verbose=True):
    """Performs the deflection by splitting the full latitude range into distinct bands which are done one at a time.

        See *_lens_gcband_sym* for the single band (north and south hemispheres) lensing.

    """
    assert spin >= 0,spin
    times = utils.timer(verbose, suffix=' ' + __name__)
    target_nt = 3 ** 1 * 2 ** (11 + facres) # on one hemisphere

    #co-latitudes
    th1s = np.arange(nband) * (np.pi * 0.5 / nband)
    th2s = np.concatenate((th1s[1:],[np.pi * 0.5]))
#   Uncomment for testing
#    print('cut test of [theta1, theta2]=[pi/3, pi*2/3]')
#    print('Nband=%i'%nband)
#    th1s = np.pi/6
#    th2s = np.pi/3
    nt_perband = int(target_nt / nband)
    print('nt_perband=%i'%nt_perband)
    if np.iscomplexobj(dlm): # inputs are the spin 1 d map
        lmax = hp.Alm.getlmax(dlm.size)
        redtot, imdtot = hp.alm2map_spin([dlm, np.zeros_like(dlm) if dclm is None else dclm], nside, 1, lmax)
    else:
        assert dclm is not None and dclm.size == dlm.size and dlm.size == hp.nside2npix(nside)
        redtot = dlm
        imdtot = dclm
    times.add('defl. spin 1 transform')
    interp_pix = 0
    ret = np.empty(hp.nside2npix(nside),dtype = float if spin == 0 else complex)
    for ib, th1, th2 in zip(range(nband), th1s, th2s):
#   Uncomment for testing
#    for ib in range(1):
#        print('ib=',ib)
#        th1 = th1s
#        th2=th2s
        if verbose: print("BAND %s in %s :"%(ib, nband))
        pixn = hp.query_strip(nside, th1, th2, inclusive=True)
        pixs = hp.query_strip(nside, np.pi- th2,np.pi - th1, inclusive=True)
        thtp, phipn = angles.get_angles(nside, pixn, redtot[pixn], imdtot[pixn], 'north', verbose=verbose)
        thtps, phips = angles.get_angles(nside, pixs, redtot[pixs], imdtot[pixs], 'south', verbose=verbose)

        # Adding a 10 pixels buffer for new angles to be safely inside interval.
        # th1,th2 is mapped onto pi - th2,pi -th1 so we need to make sure to cover both buffers
        mathtp = np.max(thtp); mithtp = np.min(thtp)
        mathtps = np.max(thtps); mithtps = np.min(thtps)
        buffN = 10 * (mathtp - mithtp) / (nt_perband - 1) / (1. - 2. * 10. / (nt_perband - 1))
        buffS = 10 * (mathtps - mithtps) / (nt_perband - 1) / (1. - 2. * 10. / (nt_perband - 1))
        th1 = min(np.pi - (mathtps + buffS),mithtp - buffN)
        th2 = max(np.pi - (mithtps - buffS),mathtp + buffN)

        #==== these are the theta and limits. It is ok to go negative or > 180
        if verbose: print('input t1,t2 %.3f %.3f in degrees'%(th1 /np.pi * 180,th2/np.pi * 180.))
        if verbose: print('North %.3f and South %.3f buffers in amin'%(buffN /np.pi * 180 * 60,buffS/np.pi * 180. * 60.))
        nphi = utils.get_nphi(th1, th2, facres=facres)
        dphi_patch = (2. * np.pi) / nphi * max(np.sin(th1),np.sin(th2))
        dth_patch = (th2 - th1) / (nt_perband -1)
        if verbose: print("cell (theta,phi) in amin (%.3f,%.3f)" % (dth_patch / np.pi * 60. * 180, dphi_patch / np.pi * 60. * 180))
        times.add('defl. angles calc.')
        len_nr, len_ni, len_sr, len_si = _lens_gcband_sym(spin, glm, th1, th2, nt_perband, nphi, thtp, phipn, thtps, phips,
                                                         clm=clm, times=times)
        if spin == 0:
            ret[pixn] = len_nr
            ret[pixs] = len_sr
        else :
            ret[pixn] = (len_nr + 1j * len_ni) * angles.rotation(nside, spin, pixn, redtot[pixn], imdtot[pixn])
            ret[pixs] = (len_sr + 1j * len_si) * angles.rotation(nside, spin, pixs, redtot[pixs], imdtot[pixs])
            times.add(r'pol. //-transport rot.')
        interp_pix += nphi * nt_perband * 2
    if verbose: print(times)
    if verbose: print(r"Number of interpolating pixels: %s ~ %s^2 "%(interp_pix, int(np.sqrt(interp_pix))))
    return ret

### Modified by FG
def _strip_lens_gclm_sym_timed(spin, dlm, glm, nside, theta1, theta2, nband=4, facres=0, clm=None, dclm=None, verbose=True):
    """Performs the deflection by splitting the full latitude range into distinct bands which are done one at a time.

        See *_lens_gcband_sym* for the single band (north and south hemispheres) lensing.

    """
    assert spin >= 0,spin
    times = utils.timer(verbose, suffix=' ' + __name__)
    target_nt = 3 ** 1 * 2 ** (11 + facres) # on one hemisphere

    #co-latitudes
    th1s = np.arange(nband) * ((theta2 - theta1) / nband) + theta1
    th2s = np.concatenate((th1s[1:],[theta2]))
    print('th1=%.5f, th2=%.5f'%(th1s[0], th2s[0]))
    nt_perband = int(2 * target_nt * (theta2 - theta1) / np.pi / nband)
    print('nt_perband=%i'%nt_perband)
    if np.iscomplexobj(dlm): # inputs are the spin 1 d map
        lmax = hp.Alm.getlmax(dlm.size)
        redtot, imdtot = hp.alm2map_spin([dlm, np.zeros_like(dlm) if dclm is None else dclm], nside, 1, lmax)
    else:
        assert dclm is not None and dclm.size == dlm.size and dlm.size == hp.nside2npix(nside)
        redtot = dlm
        imdtot = dclm
    times.add('defl. spin 1 transform')
    interp_pix = 0
    ret = np.empty(hp.nside2npix(nside),dtype = float if spin == 0 else complex)
    for ib, th1, th2 in zip(range(nband), th1s, th2s):

#    for ib in range(1):
        print('th1=%.5f, th2=%.5f'%(th1, th2))

        if verbose: print("BAND %s in %s :"%(ib, nband))
        
        ## For MUSE3G, only the southern hemisphere is used. 
        ## So just get rid of the settings for northern hemisphere.
        #pixn = hp.query_strip(nside, np.pi - th2, np.pi - th1, inclusive=True)
        pixs = hp.query_strip(nside, th1, th2, inclusive=True)
        #thtp, phipn = angles.get_angles(nside, pixn, redtot[pixn], imdtot[pixn], 'north', verbose=verbose)
        thtps, phips = angles.get_angles(nside, pixs, redtot[pixs], imdtot[pixs], 'south', verbose=verbose)

        # Adding a 10 pixels buffer for new angles to be safely inside interval.
        # th1,th2 is mapped onto pi - th2,pi -th1 so we need to make sure to cover both buffers
        #mathtp = np.max(thtp); mithtp = np.min(thtp)
        #print('mathtp=%.5f, mithtp=%.5f'%(mathtp, mithtp))
        mathtps = np.max(thtps); mithtps = np.min(thtps)
        print('mathtps=%.5f, mithtps=%.5f'%(mathtps, mithtps))
        #buffN = 10 * (mathtp - mithtp) / (nt_perband - 1) / (1. - 2. * 10. / (nt_perband - 1))
        buffS = 10 * (mathtps - mithtps) / (nt_perband - 1) / (1. - 2. * 10. / (nt_perband - 1))
        #th1 = min(np.pi - (mathtps + buffS),mithtp - buffN)
        #th2 = max(np.pi - (mithtps - buffS),mathtp + buffN)
        th1 = mithtps - buffS
        th2 = mathtps + buffS

        #==== these are the theta and limits. It is ok to go negative or > 180
        if verbose: print('input t1,t2 %.3f %.3f in degrees'%(th1 /np.pi * 180,th2/np.pi * 180.))
        if verbose: print('South %.3f buffers in amin'%(buffS/np.pi * 180. * 60.))
        nphi = utils.get_nphi(th1, th2, facres=facres)
        dphi_patch = (2. * np.pi) / nphi * max(np.sin(th1),np.sin(th2))
        dth_patch = (th2 - th1) / (nt_perband -1)
        if verbose: print("cell (theta,phi) in amin (%.3f,%.3f)" % (dth_patch / np.pi * 60. * 180, dphi_patch / np.pi * 60. * 180))
        times.add('defl. angles calc.')
#        len_nr, len_ni, len_sr, len_si = _lens_gcband_sym(spin, glm, np.pi-th2, np.pi-th1, nt_perband, nphi, thtp, phipn, thtps, phips,
#                                                         clm=clm, times=times)

        tgrid = utils.thgrid(np.pi - th2, np.pi - th1).mktgrid(nt_perband)
        if spin == 0:
            vtm = shts.glm2vtm_sym(0, utils.thgrid.th2colat(tgrid), glm)
        else:
            vtm = shts.vlm2vtm_sym(spin, utils.thgrid.th2colat(tgrid), utils.alm2vlm(glm, clm=clm))
        times.add('vtm')

        # Symmetric band on south hemisphere
        filtmap = vtm2filtmap(spin, vtm[slice(2 * nt_perband - 1, nt_perband - 1, -1)], nphi, phiflip=np.where((np.pi - tgrid) > np.pi)[0])
        t_grid = utils.thgrid(th1, th2).togridunits(thtps, nt_perband)
        p_grid = phips / ((2. * np.pi) / nphi)
        times.add('vtm2filtmap')
        lenmapsr = bicubic.deflect(np.require(filtmap.real, np.float64, requirements='F'), t_grid, p_grid)
        lenmapsi = bicubic.deflect(np.require(filtmap.imag, np.float64, requirements='F'), t_grid, p_grid) if spin > 0 else None
        times.add('interp')

        if spin == 0:
            #ret[pixn] = len_nr
            ret[pixs] = lenmapsr
        else :
            #ret[pixn] = (len_nr + 1j * len_ni) * angles.rotation(nside, spin, pixn, redtot[pixn], imdtot[pixn])
            ret[pixs] = (lenmapsr + 1j * lenmapsi) * angles.rotation(nside, spin, pixs, redtot[pixs], imdtot[pixs])
            times.add(r'pol. //-transport rot.')
        interp_pix += nphi * nt_perband * 2

    if verbose: print(times)
    if verbose: print(r"Number of interpolating pixels: %s ~ %s^2 "%(interp_pix, int(np.sqrt(interp_pix))))
    return ret


def _lens_gcband_sym(spin, glm, th1, th2, nt, nphi, thtpn, phipn, thtps, phips, clm=None, times=None):
    r"""Returns deflected maps between a co-latitude range in the North Hemisphere and its South hemisphere counterpart.

        This routine performs the deflection only, and not the rotation sourced by the local axes //-transport.

        Args:
            spin: spin of the field to deflect described by gradient glm and curl clm mode.
            glm:  gradient lm components of the field, in healpy format.
            th1:  input co-latitude range is (th1,th2) on North hemisphere and (pi - th2,pi -th1) on south hemisphere.
            th2:  input co-latitude range is (th1,th2) on North hemisphere and (pi - th2,pi -th1) on south hemisphere.
            nt:  Number of tht point to use for the interpolation.
            nphi : Number of phi points to use for the interpolation.
            thtpn: deflected colatitude on north hemisphere sector.
            phipn: deflected longitude on north hemisphere sector.
            thtps: deflected colatitude on south hemisphere sector.
            phips: deflected longitude on south hemisphere sector.

            clm(optional) : curl lm components of the field to deflect, in healpy ordering.

        Returns:
            real and imaginary parts of the north and south bands (re-n, im-n, re-s, im-s)

    """
    assert spin >= 0
    assert len(thtpn) == len(phipn) and len(thtps) == len(phips)
    if times is None: times = utils.timer(True)

    tgrid = utils.thgrid(th1, th2).mktgrid(nt)
    if spin == 0:
        vtm = shts.glm2vtm_sym(0, utils.thgrid.th2colat(tgrid), glm)
    else:
        vtm = shts.vlm2vtm_sym(spin, utils.thgrid.th2colat(tgrid), utils.alm2vlm(glm, clm=clm))
    times.add('vtm')

    # Band on north hemisphere
    filtmap = vtm2filtmap(spin, vtm[0:nt], nphi, phiflip=np.where((tgrid < 0))[0])
    t_grid = utils.thgrid(th1, th2).togridunits(thtpn, nt)
    p_grid = phipn / ((2. * np.pi) / nphi)
    times.add('vtm2filtmap')
    lenmapnr = bicubic.deflect(np.require(filtmap.real, np.float64, requirements='F'), t_grid, p_grid)
    lenmapni = bicubic.deflect(np.require(filtmap.imag, np.float64, requirements='F'), t_grid, p_grid) if spin > 0 else None
    times.add('interp')

    # Symmetric band on south hemisphere
    filtmap = vtm2filtmap(spin, vtm[slice(2 * nt - 1, nt - 1, -1)], nphi, phiflip=np.where((np.pi - tgrid) > np.pi)[0])
    t_grid = utils.thgrid(np.pi - th2, np.pi - th1).togridunits(thtps, nt)
    p_grid = phips / ((2. * np.pi) / nphi)
    times.add('vtm2filtmap')
    lenmapsr = bicubic.deflect(np.require(filtmap.real, np.float64, requirements='F'), t_grid, p_grid)
    lenmapsi = bicubic.deflect(np.require(filtmap.imag, np.float64, requirements='F'), t_grid, p_grid) if spin > 0 else None
    times.add('interp')
    return lenmapnr, lenmapni, lenmapsr, lenmapsi

def vtm2filtmap(spin, vtm, nphi, threads=None, phiflip=()):
    return shts.vtm2map(spin, vtm, nphi, pfftwthreads=threads, bicubic_prefilt=True, phiflip=phiflip)

