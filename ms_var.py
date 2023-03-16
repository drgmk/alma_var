import os
import shutil
import datetime
import logging
import aplpy
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import astropy.units as un
import astropy.io
import astropy.time
from astropy.table import QTable
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord

# package load for modular CASA
import numexpr
import casatools.ms
import casatools.msmetadata
import casatools.table
from casatasks import listobs, uvmodelfit, tclean, split, exportfits

ms = casatools.ms()
msmd = casatools.msmetadata()
tb = casatools.table()

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1

logging.basicConfig(level=logging.INFO)


def export_ms(msfilename, xcor=True, acor=False):
    """Return visibilities etc.

    Direct copy of Luca Matra's export."""

    cc = 2.9979e10  # cm/s

    # Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
    tb.open(msfilename)
    data    = tb.getcol("DATA")
    uvw     = tb.getcol("UVW")
    weight  = tb.getcol("WEIGHT")
    ant1    = tb.getcol("ANTENNA1")
    ant2    = tb.getcol("ANTENNA2")
    flags   = tb.getcol("FLAG")
    spwid   = tb.getcol("DATA_DESC_ID")
    time    = tb.getcol("TIME")
    # scan    = tb.getcol("SCAN_NUMBER")
    tb.close()

    if np.any(flags):
        logging.warning(f"{msfilename}: some of the data is FLAGGED")

    logging.info("Found data with "+str(data.shape[-1])+" uv points")
    if data.shape[-1] == 0:
        return [], [], [], [], []

    ms.open(msfilename)
    spw_info = ms.getspectralwindowinfo()
    nchan = spw_info[list(spw_info.keys())[0]]['NumChan']
    npol  = spw_info[list(spw_info.keys())[0]]['NumCorr']
    ms.close()
    logging.info("with "+str(nchan)+" channels per SPW and "+str(npol)+" polarizations,")

    # Use CASA table tools to get frequencies, which are needed to
    # calculate u-v points from baseline lengths
    tb.open(msfilename+"/SPECTRAL_WINDOW")
    freqs = tb.getcol("CHAN_FREQ")
    rfreq = tb.getcol("REF_FREQUENCY")
    tb.close()

    logging.info(str(freqs.shape[1])+" SPWs and Channel 0 frequency of 1st SPW of "+str(rfreq[0]/1e9)+" GHz")
    logging.info("corresponding to "+str(2.9979e8/rfreq[0]*1e3)+" mm")
    logging.info("Average wavelength is "+str(2.9979e8/np.average(rfreq)*1e3)+" mm")

    if np.max(uvw[0:2, :]) == 0:
        logging.info('No non-zero baselines, returning')
        return [], [], [], [], []

    logging.info("Dataset has baselines between "+str(np.min(np.sqrt(uvw[0, :]**2.0+uvw[1, :]**2.0))) +
                 " and "+str(np.max(np.sqrt(uvw[0, :]**2.0+uvw[1, :]**2.0)))+" m")

    # Initialize u and v arrays (coordinates in Fourier space)
    uu = np.zeros((freqs.shape[0], uvw[0, :].size))
    vv = np.zeros((freqs.shape[0], uvw[0, :].size))

    # Fill u and v arrays appropriately from data values.
    for i in np.arange(freqs.shape[0]):
        for j in np.arange(uvw.shape[1]):
            uu[i, j] = uvw[0, j]*freqs[i, spwid[j]]/(cc/100.0)
            vv[i, j] = uvw[1, j]*freqs[i, spwid[j]]/(cc/100.0)

    # Extract real and imaginary part of the visibilities at all u-v
    # coordinates, for both polarization states (XX and YY), extract
    # weights which correspond to 1/(uncertainty)^2
    Re_xx = data[0, :, :].real
    Re_yy = data[1, :, :].real
    Im_xx = data[0, :, :].imag
    Im_yy = data[1, :, :].imag
    weight_xx = weight[0, :]
    weight_yy = weight[1, :]

    # Since we don't care about polarization, combine polarization states
    # (average them together) and fix the weights accordingly. Also if
    # any of the two polarization states is flagged, flag the outcome of
    # the combination.
    flags = flags[0, :, :]*flags[1, :, :]
    Re = np.where((weight_xx + weight_yy) != 0, (Re_xx*weight_xx + Re_yy*weight_yy) / (weight_xx + weight_yy), 0.)
    Im = np.where((weight_xx + weight_yy) != 0, (Im_xx*weight_xx + Im_yy*weight_yy) / (weight_xx + weight_yy), 0.)
    wgts = (weight_xx + weight_yy)

    # Find which of the data represents cross-correlation between two
    # antennas as opposed to auto-correlation of a single antenna.
    # We don't care about the latter so we don't want it.
    xc = np.where(ant1 != ant2)[0]
    ac = np.where(ant1 == ant2)[0]
    if xcor:
        xc = xc
        if acor:
            xc = np.logical_or(xc, ac)
    elif acor:
        xc = ac

    # Select data
    time = time[xc]
    # scan = scan[xc]
    data_real = Re[:, xc]
    data_imag = Im[:, xc]
    flags = flags[:, xc]
    # data_wgts = wgts[xc]
    data_uu = uu[:, xc]
    data_vv = vv[:, xc]
    data_wgts = np.reshape(np.repeat(wgts[xc], uu.shape[0]), data_uu.shape)

    # Select only data that is NOT flagged, this step has the unexpected
    # effect of flattening the arrays to 1d
    data_real = data_real[np.logical_not(flags)]
    data_imag = data_imag[np.logical_not(flags)]
    # flags_ = flags[np.logical_not(flags)]
    data_wgts = data_wgts[np.logical_not(flags)]
    data_uu = data_uu[np.logical_not(flags)]
    data_vv = data_vv[np.logical_not(flags)]
    time = time[np.logical_not(flags[0])]
    # scan = scan[np.logical_not(flags[0])]

    time /= (24*60*60)  # to MJD
    vis = data_real + 1j*data_imag

    # sort into time order
    srt = np.argsort(time)
    data_uu = data_uu[srt]
    data_vv = data_vv[srt]
    vis = vis[srt]
    data_wgts = data_wgts[srt]
    time = time[srt]

    # re-weighting as suggested by Loomis+
    wgt_mean = np.mean(data_wgts)
    data_std = np.std(vis)
    rew = (1/data_std**2)/wgt_mean
    logging.info(f're-weighting value (1dof): {rew}')
    data_wgts *= rew

    return data_uu, data_vv, vis, data_wgts, time

def load_npy_vis(savefile):
    """Load visibilities, restoring reals."""
    u, v, vis, wt, time = np.load(savefile)
    return u.real, v.real, vis, wt.real, time.real

def h_filter(vis, wt):
    """Return a matched filter for the model visibilities and data weights given.

    Assumes the model is a point source, so norm and h are simplified. This
    makes things run much quicker.

    Parameters
    ----------
    vis : numpy array of complex
        Model visibilities, 1d (nvis*nptsrc) or 2d (nvis, nptsrc)
    wt : numpy array
        Data weights, 1d (nvis)
    """

    #     R_inv = np.identity(len(wt))
    norm = 1/np.sqrt(np.sum(wt))
    # if vis.shape == wt.shape:
        #         vis = vis_[:,np.newaxis]
        #         norm = 1 / np.sqrt( np.matmul(vis.conj().T, np.matmul(wt*R_inv, vis)) ).squeeze()
        #         h = norm * np.matmul(wt*R_inv, vis).squeeze()
        # h = norm * wt * vis
    # else:
        #         norm = 1 / np.sqrt(np.einsum('ij,ij->j', vis_.conj(), np.matmul(wt*R_inv, vis_)) )
        #         h = norm * np.matmul(wt*R_inv, vis_)
    h = norm * wt[:, np.newaxis] * vis

    return h, norm


def uv_shift(u, v, ra, dec, flatxy=True):
    """Return visibilities of a point source at some ra/dec offset.

    Most of the time is in the exponential, and
    fiddling the outer product doesn't seem to help.

    Parameters
    ----------
    u, v : numpy array
        Points in u,v space (nvis)
    ra,dec : numpy array
        Offsets of points from phase center (1 or 2d)
    flatxy : bool, optional
        Return visibilities as flattened array.
    """
    out = np.outer(u, -ra) + np.outer(v, dec)
    arg = -2*np.pi*1j*out
    a = numexpr.evaluate('exp(arg)')
    # a = np.exp(arg)
    if flatxy:
        return a.reshape((-1,))
    else:
        return a.reshape((-1,)+ra.shape)


def mjd2date(d):
    t0 = datetime.datetime(1, 1, 1, 12)
    dt = datetime.timedelta(2400000.5 + d - 1721426.0)
    return t0 + dt


def plot_fits_sources(fits, ra, dec):
    """Plot sources on a FITS image.

    Multiple hacks needed...
    """

    # fix for UTC in fits file, but astropy wants utc
    h = astropy.io.fits.open(fits)
    # hack for UTC in file but astropy wants utc
    h[0].header['TIMESYS'] = h[0].header['TIMESYS'].lower()

    fig = aplpy.FITSFigure(h[0])
    fig.show_colorscale(stretch='linear', cmap='viridis')
    s = SkyCoord(ra, dec)
    fig.show_markers(s.ra, s.dec, marker='o', edgecolor='white')
    # hack for add beam to work
    b = aplpy.Beam(fig)
    b._wcs = b._wcs[0]
    b._wcs.is_celestial = True
    b._wcs.pixel_scale_matrix = fig._wcs.pixel_scale_matrix
    b.show(major=h[0].header['BMAJ'],
           minor=h[0].header['BMIN'],
           angle=h[0].header['BPA'],
           facecolor='white', edgecolor='black')
    b.set_hatch('/')
    fig.set_title(fits.replace('.fits', ''))
    fig.savefig(fits.replace('.fits', '.png'))


def get_ms_info(msfilepath):
    """Get various pieces of info from an ms.

    Primary beam, Technical Handbook S10.3.1

    Resolution approximation from:
    https://almascience.eso.org/about-alma/alma-basics
    """
    tb.open(msfilepath)
    uvw = tb.getcol("UVW")
    times = tb.getcol("TIME") / (24*60*60)
    tb.close()

    msmd.open(msfilepath)
    diams = msmd.antennadiameter()
    spws = msmd.spwsforscans()
    spws = np.unique(np.array([spws[k] for k in spws.keys()]).flatten())

    diam = np.mean([diams[k]['value'] for k in diams.keys()]) \
        * un.Unit(diams['0']['unit'])
    freqs = [msmd.reffreq(s) for s in spws]
    freqs = [k['m0']['value'] for k in freqs]*un.Unit(freqs[0]['m0']['unit'])
    meanfreq = np.mean(freqs)
    meanwav = np.mean(3e8*un.m/un.s / freqs).to(un.m)
    logging.info(f'mean freq {meanfreq}, mean wav {meanwav}')

    pb_hwhm = (1.13 * meanwav / diam / 2 * un.rad).to(un.arcsec)
    max_b_km = np.max(np.sqrt(uvw[0, :]**2 + uvw[1, :]**2)) / 1e3
    if max_b_km == 0:
        res = np.inf * un.arcsec
    else:
        res = 76 / max_b_km / meanfreq.to('GHz').value * un.arcsec

    field_id = np.unique([msmd.fieldsforspw(s) for s in spws])
    if len(field_id) > 1:
        logging.warning(f'{len(field_id)} fields for spws {spws}')
    field_id = field_id[0]
    field_name = msmd.namesforfields(field_id)[0]

    out = msmd.phasecenter(fieldid=field_id)
    ra = out['m0']['value'] * un.Unit(out['m0']['unit'])
    dec = out['m1']['value'] * un.Unit(out['m1']['unit'])

    meantime = mjd2date(np.mean(times))
    msmd.close()

    info = {'phase_center': [ra, dec],
            'pb_hwhm': pb_hwhm,
            'spatial_res': res,
            'mean_time': meantime,
            'field_id': field_id,
            'field_name': field_name}

    return info


def get_gaia_offsets(ra, dec, radius, date, min_plx_mas=None):
    """Return offsets of Gaia sources from ra/dec in radians."""
    coord = SkyCoord(ra=ra, dec=dec, frame='icrs')
    print(f'get_gaia_offsets: search at {ra}, {dec}')

    r = Gaia.query_object_async(coordinate=coord, radius=radius)
    r = QTable(r)
    r.sort('dist')

    if min_plx_mas:
        r = r[r['parallax'] > min_plx_mas * un.mas]

    # proper motion correction
    date_ = astropy.time.Time(date)
    r['ra_ep'] = r['ra'] + (date_.byear-2016.0)*un.year * r['pmra'] / np.cos(r['dec'])
    r['dec_ep'] = r['dec'] + (date_.byear-2016.0)*un.year * r['pmdec']
    nopm = r['pmra'].mask == True
    r['ra_ep'][nopm] = r['ra'][nopm]
    r['dec_ep'][nopm] = r['dec'][nopm]

    # convert to offsets
    ra_off = np.array((r['ra_ep']- ra).to(un.rad).value)
    dec_off = np.array((r['dec_ep'] - dec).to(un.rad).value)
    ra_off = (ra_off + np.pi) % (2 * np.pi) - np.pi
    dec_off = (dec_off + np.pi) % (2 * np.pi) - np.pi

    return ra_off, dec_off, r


def smooth_plot(times, v_in, det_snr, show_sig=True,
                ylab='$|Sum(V.w)|$',
                outdir='.', outpre='', outfile=f'vis_time_smooth.png'):
    """Diagnostic plot for time series, return True for detection."""
    tplot2 = (times-np.min(times))*24*60
    dt = np.median(np.diff(times))*24*60*60  # dt in seconds
    nw = 60
    if len(times) < nw/2:
        nw = len(times)//2
    ws = np.arange(nw)+1
    T = np.zeros((nw, len(times)))
    wpk = []
    pk = []
    snr = []

    vmin = 1e5
    for i, wi in enumerate(ws):
        conv = np.convolve(v_in, np.repeat(1, wi)/wi, mode='valid')
        conv = (conv-np.mean(conv))*np.sqrt(wi) + np.mean(conv)
        T[i, (wi-1)//2:len(v_in)-wi//2] = conv

        # find significant +ve outliers
        clipped, _, _ = scipy.stats.sigmaclip(conv)
        mn, std = np.mean(clipped), np.std(clipped)
        ok = np.where(T[i] > mn + det_snr*std)[0]
        for o in ok:
            wpk.append(wi)
            pk.append(tplot2[o])
            snr.append((T[i, o]-mn)/std)

        if np.min(conv) < vmin:
            vmin = np.min(conv)

    wpk = np.array(wpk)
    pk = np.array(pk)
    snr = np.array(snr)

    fig, ax = plt.subplots(2, figsize=(8, 6), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})
    ax[0].imshow(T, aspect='auto', origin='lower', vmin=vmin,
                 extent=(np.min(tplot2), np.max(tplot2),
                         np.min(ws)-0.5, np.max(ws)+0.5))
    if show_sig and len(snr) > 0:
        mx = np.argmax(snr)
        ax[0].plot(pk[mx], wpk[mx], '+k')
    ax[1].plot(tplot2, v_in, '.', label='$|Sum(V.w)|$')
    ax[1].set_xlabel('Time / minutes')
    ax[1].set_ylabel(ylab)
    ax[0].set_ylabel(f'smoothing width / $\Delta t$={dt:3.2f} seconds')
    fig.tight_layout()
    figname = f'{outdir}/{outpre}{outfile}'
    fig.savefig(figname)
    plt.close(fig)

    return len(wpk) > 0, figname


def summed_search(savefile, det_snr, reloutdir='.', outpre='',
                  plot_vis_hist=False, plot_raw_vis=False):
    """Point source position-free search.

    Parameters
    ----------
    savefile : str
        Name of numpy savefile with visibilities.
    det_snr : float
        SNR threshold for detection flagging.
    reloutdir : str
        Relative location from savefile in which to put output plots.
    outpre : str, optional
        String to prepend to filename.
    plot_vis_hist : bool, optional
        Plot histogram of visibilities as sanity check.
    plot_raw_vis : bool, optional
        Plot raw visibilities as sanity check.
    """

    logging.info(f'summed search: loading from {savefile}')
    u, v, vis, wt, time = load_npy_vis(savefile)

    # unique returns sorted times
    times = np.unique(time)

    v_abs = []
    for t in times:
        ok = time == t
        v_abs.append(np.dot(np.abs(vis[ok]), np.sqrt(wt[ok])))

    v_abs = np.array(v_abs)

    # check visibility weights sensible
    # multiply by sqrt(2) assuming Re and Im independent
    if plot_vis_hist:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
        _ = ax[0].hist(np.sqrt(2)*vis.real*np.sqrt(wt), bins=100, density=True, label='Real')
        _ = ax[1].hist(np.sqrt(2)*vis.imag*np.sqrt(wt), bins=100, density=True, label='Imag')
        x = np.linspace(-3, 3)
        for a in ax:
            a.plot(x, np.max(_[0])*np.exp(-(x**2)/2))
            a.set_xlabel('snr per visibility')
            a.legend()
        ax[0].set_ylabel('density')
        fig.savefig(f'{reloutdir}/{outpre}_vis_snr.png')
        plt.close(fig)

    # look at raw and summed data
    if plot_raw_vis:
        tplot1 = (time-np.min(time))*24*60
        fig, ax = plt.subplots(2, sharex=True, figsize=(8, 4))
        ax[0].plot(tplot1, vis.real, '.', label='Real', markersize=0.3)
        ax[1].plot(tplot1, vis.real, '.', label='Imag', markersize=0.3)
        ax[1].set_xlabel('Time / minutes')
        ax[0].set_ylabel('Real')
        ax[1].set_ylabel('Imag')
        fig.tight_layout()
        fig.savefig(f'{reloutdir}/{outpre}_rawvis_time.png')
        plt.close(fig)

    # smooth light curve and plot
    outpath = f'{os.path.dirname(savefile)}/{reloutdir}'
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    return smooth_plot(times, v_abs, det_snr,
                       outdir=f'{outpath}', outfile=f'{outpre}_sum.png')


def matchedfilter_search(savefile, det_snr, ra_off=None, dec_off=None,
                         reloutdir='matchf', outpre=''):
    """Run matched filter search on saved set of visibilities.

    Parameters
    ----------
    savefile : str
        Name of numpy savefile with visibilities.
    det_snr : float
        SNR threshold for detection flagging.
    ra_off, dec_off : numpy array, list, tuple
        Coordinates for search in radians
    reloutdir : str
        Relative location from savefile in which to put output plots.
    outpre : str, optional
        String to prepend to filename.
    """

    logging.info(f'matched filter: loading from {savefile}')
    u, v, vis, wt, time = load_npy_vis(savefile)

    if ra_off is None:
        ra_off = [0.]
        dec_off = [0.]

    # get ra/dec into 1d arrays of positions
    ra = np.array(ra_off).flatten()
    dec = np.array(dec_off).flatten()

    times = np.unique(time)
    v_pos = []

    for t in times:

        ok = time == t
        vis_mod = uv_shift(u[ok], v[ok], ra, dec)
        h, _ = h_filter(vis_mod, wt[ok])
        v_pos.append(np.sqrt(2) * np.real(np.dot(vis[ok], h.conj())))

    v_pos = np.array(v_pos)

    # plots
    outpath = f'{os.path.dirname(savefile)}/{reloutdir}'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    dets = []
    fns = []
    for i in range(len(ra)):
        det, fn = smooth_plot(times, v_pos[:, i], det_snr,
                              outdir=outpath, outfile='.png', ylab='SNR',
                              outpre=f'{outpre}_{i:03d}_{np.rad2deg(ra[i])*3600:.3f}_{np.rad2deg(dec[i])*3600:.3f}.')
        dets.append(det)
        fns.append(fn)

    return dets, fns


def uvmodelfit_search(msfilepath, ra_off=None, dec_off=None, dt=None,
                      reloutdir='uvmfit', make_fits=False,
                      cleanpar=None):
    """Run uvmodelfit search on a single-scan ms."""

    msfilepath = msfilepath.rstrip('/')

    if ra_off is None:
        ra_off = [0.]
        dec_off = [0.]

    # check ms has some non-zero baselines (xcorrelation)
    ms.open(msfilepath)
    uvw = ms.getdata('UVW')
    ms.close()
    if np.max(uvw['uvw'][0]) == 0:
        logging.info(f'no non-zero baselines, returning')
        return

    if type(ra_off) in [list, np.ndarray, tuple]:
        for ra, dec in zip(ra_off, dec_off):
            uvmodelfit_search(msfilepath, ra, dec, dt=dt, reloutdir=reloutdir, make_fits=make_fits)
        return

    outdir = f"{os.path.dirname(msfilepath)}/{reloutdir.rstrip('/')}"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    wdir = outdir
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    if os.path.exists(f'{wdir}/tmp.cl'):
        shutil.rmtree(f'{wdir}/tmp.cl')

    coords = f'{ra_off:.3f}_{dec_off:.3f}'

    logging.info(f'running uv search for {os.path.basename(msfilepath)}')

    flux = []
    time = []

    out = listobs(msfilepath)

    for k in out.keys():
        if 'scan' in k:
            if dt is None:
                dt = out[k]['0']['IntegrationTime']

            t0 = out[k]['0']['BeginTime']
            t1 = out[k]['0']['EndTime']
            ts = np.arange(t0, t1, dt/(24*60*60))

    cleandir = f'{wdir}/cleans_{coords}'
    if not os.path.exists(cleandir) and make_fits:
        os.mkdir(cleandir)

    for i, t in enumerate(ts[:-1]):

        t0_ = mjd2date(t)
        t1_ = mjd2date(ts[i+1])
        t0_str = t0_.strftime('%H:%M:%S') + t0_.strftime(".%f")[:2]
        t1_str = t1_.strftime('%H:%M:%S') + t1_.strftime(".%f")[:2]
        tmid = (t + ts[i+1])/2
        time.append(tmid)

        # uvmodelfit, flux is saved in cl as complex
        if cleanpar is None:
            cleanpar = {'cell': '0.5arcsec', 'imsize': [256, 256]}
        uvmodelfit(vis=msfilepath, timerange=f"{t0_str}~{t1_str}",
                   comptype='P',
                   sourcepar=[1, ra_off, dec_off], varypar=[True, False, False],
                   outfile=f'{wdir}/tmp.cl')
        tb.open(f'{wdir}/tmp.cl')
        flux.append(np.real(tb.getcol('Flux')[0][0]))
        tb.close()

        if make_fits:
            os.system(f'rm -rf {wdir}/tmpimage*')
            tclean(vis=f'{msfilepath}', imagename=f'{wdir}/tmpimage',
                   **cleanpar, interactive=False, niter=0,
                   timerange=f"{t0_str}~{t1_str}")
            exportfits(imagename=f'{wdir}/tmpimage.image',
                       fitsimage=f'{cleandir}/{i:04d}.fits')

    tplot = (time-np.min(time))*24*60

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tplot, flux, '.')
    ax.set_xlabel('Time / minutes')
    ax.set_ylabel('Flux / Jy')
    fig.savefig(f'{outdir}/{coords}_flux_time.png')
    plt.close(fig)
    np.save(f'{outdir}/{coords}_timeflux.npy', np.vstack((time, flux)))

    shutil.rmtree(f'{wdir}/tmp.cl')
    if make_fits:
        os.system(f'rm -rf {wdir}/tmpimage*')


class AlmaVar:

    def __init__(self, ms_in, outdir=None, det_snr=4,
                 pb_factor=1.6, auto_load=True):
        """Initialise by creating output folder and averaging ms.

        Parameters
        ----------
        ms_in : str
            Path of ms file to process.
        outdir : str, optional
            Folder in which to put output (subfolder will be created here).
        det_snr : float, optional
            SNR threshold for flagging a detection.
        auto_load : bool, optional
            Automatically load what has been processed already.
            May cause problems if settings have changed.
        pb_factor : float, optional
            Number of primary beam FWHM for target search/clean images.
            1.6 gives images out to tclean default of pblimit=0.2.
        """
        # set up some output folders
        self.ms_in = ms_in.rstrip('/')
        self.ms_in_name = os.path.basename(self.ms_in)
        if outdir is None:
            outdir = os.path.dirname(ms_in)
        outdir = outdir.rstrip('/')
        self.wdir = f'{outdir}/{self.ms_in_name}.var'
        self.scandir = f'{self.wdir}/scans'
        self.ms_avg = f'{self.wdir}/{self.ms_in_name}.avg'

        if not os.path.exists(self.wdir):
            logging.info(f'creating output folder {self.wdir}')
            os.mkdir(self.wdir)

        if not os.path.exists(self.scandir):
            os.mkdir(self.scandir)

        # info from ms_in we can grab now
        ms.open(self.ms_in)
        self.ms_in_spw_info = ms.getspectralwindowinfo()
        ms.close()

        tb.open(self.ms_in)
        # check for datacolumn
        self.ms_in_datacol = 'CORRECTED'
        if 'CORRECTED' not in tb.colnames():
            self.ms_in_datacol = 'DATA'

        # some things (that might get filled later)
        self.pb_factor = pb_factor
        self.det_snr = det_snr
        self.field_gaia = {}
        self.scan_gaia = None
        self.scan_info = None
        self.scan_vis = None

        # auto load
        if auto_load:
            if os.path.exists(self.ms_avg):
                self.avg_ms_in()

    def process(self):
        """Shortcut."""
        self.avg_ms_in()
        self.split_scans(keep_scan_ms=True)
        self.clean_images()
        self.split_scans(keep_scan_ms=False)  # delete ms files
        self.summed_filter()
        self.gaia_matched_filter(min_plx_mas=1)

    def avg_ms_in(self, nchan_spw=1, spw_include=None):
        """Average ms_in down to fewer channels per spw."""

        # details, so we can exclude most data
        # TDM/FDM corresponds to FULL_RES (not SQLD or CH_AVG)
        # but we allow for various choices anyway
        if spw_include is None:
            spw_include = {'tfdm': True,
                           'sqld': False, 'chavg': False}
        msmd.open(self.ms_in)
        spws = {}
        if spw_include['tfdm']:
            spws['tfdm'] = msmd.almaspws(tdm=True, fdm=True)
        if spw_include['sqld']:
            spws['sqld'] = msmd.almaspws(sqld=True)
        if spw_include['chavg']:
            spws['chavg'] = msmd.almaspws(chavg=True)
        logging.info(f'spws: {spws}')
        msmd.close()

        # average FULL_RES down to fewer channels per spw
        # output is in DATA column
        spw_list = []
        avg_list = []
        for k in self.ms_in_spw_info.keys():
            spwid = self.ms_in_spw_info[k]['SpectralWindowId']
            if spwid in spws['tfdm']:
                spw_list.append(spwid)
                avg_list.append(self.ms_in_spw_info[k]['NumChan']//nchan_spw)

        if not os.path.exists(self.ms_avg):
            logging.info(f'averaging {self.ms_in}, using {self.ms_in_datacol}')
            logging.info(f'keeping spws:{spw_list}, widths:{avg_list}')
            split(vis=self.ms_in, outputvis=self.ms_avg, keepflags=False,
                  spw=','.join([str(s) for s in spw_list]), width=avg_list,
                  datacolumn=self.ms_in_datacol)

        # now fill some info for averaged ms
        ms.open(self.ms_avg)
        self.ms_avg_scan_info = ms.getscansummary()
        ms.close()

        # get spws in averaged data
        msmd.open(self.ms_avg)
        self.ms_avg_spws = {}
        if spw_include['tfdm']:
            self.ms_avg_spws['tfdm'] = msmd.almaspws(tdm=True, fdm=True)
        if spw_include['sqld']:
            self.ms_avg_spws['sqld'] = msmd.almaspws(sqld=True)
        if spw_include['chavg']:
            self.ms_avg_spws['chavg'] = msmd.almaspws(chavg=True)
        msmd.close()

    def split_scans(self, keep_scan_ms=False, load_scan_vis=False):
        """Split scans from averaged ms."""
        scans_sorted = [int(s) for s in self.ms_avg_scan_info.keys()]
        scans_sorted.sort()
        self.scan_info = {}
        self.scan_vis = {}
        for scan_no in scans_sorted:

            scan_no_dir = f'{self.scandir}/{scan_no}'
            if not os.path.exists(scan_no_dir):
                os.mkdir(scan_no_dir)

            # process by spw type
            for spw in self.ms_avg_spws.keys():

                if len(self.ms_avg_spws[spw]) == 0:
                    continue

                # T/FDM are named just for the scan, since this is what
                # we assume will be used by default
                if spw == 'tfdm':
                    scan_str = f'scan_{int(scan_no):02d}'
                else:
                    scan_str = f'scan_{int(scan_no):02d}_{spw}'
                scan_avg_vis = f'{scan_no_dir}/{scan_str}-vis.npy'
                scan_avg_info = scan_avg_vis.replace('-vis', '-info')
                scan_avg_ms = f'{scan_no_dir}/{scan_str}.ms'

                if os.path.exists(scan_avg_vis) and not keep_scan_ms:
                    u, v, vis, wt, time = load_npy_vis(scan_avg_vis)
                    self.scan_info[scan_no] = np.load(scan_avg_info, allow_pickle=True).item()
                else:
                    logging.info(f'splitting scan {scan_no} from {scan_avg_ms}')

                    if not os.path.exists(scan_avg_ms):
                        split(vis=self.ms_avg, outputvis=scan_avg_ms,
                              scan=scan_no, datacolumn='DATA',
                              spw=','.join([str(s) for s in self.ms_avg_spws[spw]]))

                    # get data from ms
                    u, v, vis, wt, time = export_ms(scan_avg_ms)
                    # save, this will make everything complex
                    np.save(scan_avg_vis, np.array([u, v, vis, wt, time]))

                    info = get_ms_info(scan_avg_ms)
                    info['nvis'] = len(vis)
                    info['scan_str'] = scan_str
                    info['scan_dir'] = scan_no_dir
                    info['spw'] = spw
                    info['scan_avg_ms'] = scan_avg_ms
                    info['scan_avg_vis'] = scan_avg_vis
                    self.scan_info[scan_no] = info
                    # info has Quantities so json doesn't work
                    # with open(scan_avg_info, 'w') as fh:
                    #     json.dump(info, fh)
                    np.save(scan_avg_info, info)

                if load_scan_vis:
                    self.scan_vis[scan_str] = {'u': u, 'v': v, 'vis': vis, 'wt': wt, 'time': time}

                if len(vis) == 0:
                    logging.info(f'{scan_no}: no visibilities')

                if not keep_scan_ms and os.path.exists(scan_avg_ms):
                    shutil.rmtree(scan_avg_ms)

    def clean_images(self, scans=None, oversample=3, min_size=256, overwrite=False):
        """Create tclean images for specified scans.

        Parameters
        ----------
        scans : list, optional
            List of scans to image, default is all scans
        oversample : int
            Factor to oversample PSF in images (pixel size = res / 2 / oversample).
        min_size : int
            Minimum x/y size for image.
        overwrite : bool, optional
            Overwrite existing FITS files.
        """
        if scans is None:
            scans = [s for s in self.scan_info.keys()]

        for scan in scans:

            # check we can make an image (e.g. not for zero baselines)
            if not np.isfinite(self.scan_info[scan]['spatial_res']):
                logging.info(f'skipping scan {scan}, no baselines')
                continue

            # get a sensible looking image. pblimit is by default 0.2, but fwhm is 0.5,
            pix_sz = self.scan_info[scan]['spatial_res'].to(un.arcsec).value / 2 / oversample
            img_fov = self.scan_info[scan]['pb_hwhm'].to(un.arcsec).value * 2 * self.pb_factor
            img_sz = int(img_fov / pix_sz)
            if img_sz < min_size:
                img_sz = min_size
                pix_sz = img_fov / img_sz

            outfits = (f"{self.scan_info[scan]['scan_dir']}/"
                       f"{self.scan_info[scan]['scan_str']}.fits")

            if not overwrite and os.path.exists(outfits):
                continue

            logging.info(f'making image {outfits} with cell:{pix_sz}, size:{img_sz}')

            if overwrite and os.path.exists(outfits):
                os.unlink(outfits)

            tclean(vis=f"{self.scan_info[scan]['scan_avg_ms']}",
                   imagename=f'{self.wdir}/tmpimage',
                   cell=f'{pix_sz}arcsec', imsize=[img_sz, img_sz],
                   interactive=False, niter=0)
            exportfits(imagename=f'{self.wdir}/tmpimage.image',
                       fitsimage=outfits)
            os.system(f'rm -rf {self.wdir}/tmpimage*')

    def link_to_field(self, file, scan):
        """Symlink a file in a field directory."""
        fn = self.scan_info[scan]['field_name']

        fdir = f'{self.wdir}/{fn}'
        if not os.path.exists(fdir):
            os.mkdir(fdir)

        link = f'{fdir}/{scan:03d}_{os.path.basename(file)}'
        link_rel = os.path.relpath(os.path.dirname(file), fdir)
        linked = f'{link_rel}/{os.path.basename(file)}'
        if not os.path.exists(link):
            os.symlink(linked, link)

    def summed_filter(self, scans=None):
        """Summed search for specified scans."""
        if scans is None:
            scans = [s for s in self.scan_info.keys()]

        for scan in scans:
            if self.scan_info[scan]['nvis'] > 0:
                det, fn = summed_search(self.scan_info[scan]['scan_avg_vis'],
                                        self.det_snr,
                                        outpre=self.scan_info[scan]['scan_str'])
                if det:
                    self.link_to_field(fn, scan)

    def matched_filter(self, scans=None, ra_off=None, dec_off=None):
        """Matched filter search for specified scans."""
        if scans is None:
            scans = [s for s in self.scan_info.keys()]

        if ra_off is None:
            ra_off = [0.]
            dec_off = [0.]

        for scan in scans:
            if self.scan_info[scan]['nvis'] > 0:
                dets, fns = matchedfilter_search(self.scan_info[scan]['scan_avg_vis'],
                                                 self.det_snr,
                                                 outpre=self.scan_info[scan]['scan_str'],
                                                 ra_off=ra_off, dec_off=dec_off)
                for det, fn in zip(dets, fns):
                    if det:
                        self.link_to_field(fn, scan)

    def gaia_sources(self, scans=None, min_plx_mas=None, update=False):
        """Find and save Gaia DR3 sources in FOV for specified scans.

        Sources are specified per field, which might be observed
        multiple times, so more efficient to save per field.
        """
        if scans is None:
            scans = [s for s in self.scan_info.keys()]

        for scan in scans:
            field = self.scan_info[scan]['field_id']
            if field not in self.field_gaia.keys() or update:
                ra_off, dec_off, r = get_gaia_offsets(self.scan_info[scan]['phase_center'][0],
                                                      self.scan_info[scan]['phase_center'][1],
                                                      self.scan_info[scan]['pb_hwhm'] * self.pb_factor,
                                                      self.scan_info[scan]['mean_time'],
                                                      min_plx_mas=min_plx_mas)
                self.field_gaia[field] = {}
                self.field_gaia[field]['ra_off'] = ra_off
                self.field_gaia[field]['dec_off'] = dec_off
                self.field_gaia[field]['table'] = r

                # plot with sources
                fits = (f"{self.scan_info[scan]['scan_dir']}/"
                        f"{self.scan_info[scan]['scan_str']}.fits")
                if os.path.exists(fits):
                    plot_fits_sources(fits, r['ra_ep'], r['dec_ep'])

    def gaia_matched_filter(self, scans=None, min_plx_mas=None):
        """Run matched filter for Gaia sources in FOV."""
        if scans is None:
            scans = [s for s in self.scan_info.keys()]

        for scan in scans:
            if self.scan_info[scan]['nvis'] > 0:
                field = self.scan_info[scan]['field_id']

                if field not in self.field_gaia.keys():
                    self.gaia_sources(scans=scans, min_plx_mas=min_plx_mas)

                if len(self.field_gaia[field]['ra_off']) == 0:
                    continue

                self.matched_filter(scans=[scan],
                                    ra_off=self.field_gaia[field]['ra_off'],
                                    dec_off=self.field_gaia[field]['dec_off'])
