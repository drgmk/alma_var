import os
import shutil
import datetime
import logging
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# test what we are using and import appropriately
try:
    __casashell_state__
    print('Assuming packaged CASA')
    logging.warning('many things will not work')
    NUMEXPR = False
except NameError:
    print('Assuming modular CASA')
    import numexpr
    NUMEXPR = True
    import aplpy
    import astropy.units as un
    import astropy.io
    import astropy.time
    from astropy.table import QTable
    from astroquery.gaia import Gaia
    from astropy.coordinates import SkyCoord

    # set up logger here, need to pre-empt implicit starting
    # of logger otherwise with casatasks import
    import casatools.logsink
    casalog = casatools.logsink('/tmp/casa.log')
    casalog.setglobal()
    import casatools.ms
    import casatools.msmetadata
    import casatools.table
    from casatasks import listobs, uvmodelfit, tclean, split, exportfits, ft, uvsub, importfits

    ms = casatools.ms()
    msmd = casatools.msmetadata()
    tb = casatools.table()

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1

    # suppress WCS and other info/warnings
    logging.getLogger('astroquery').setLevel(logging.ERROR)
    logging.getLogger('astropy').setLevel(logging.WARNING)


def parallel_ms_in(ms_in, outdir, fits_in=None):
    """Helper to run multiprocessing, use with pool.starmap.

    Run with commands like:
    import alma_var
    import glob
    import os
    import multiprocessing
    fs = glob.glob('/Users/grant/astro/data/alma/arks/*/visibilities/*fav.cor*ms')
    fits = [glob.glob(f'{os.path.dirname(f)}/../images/*model.fits')[0] for f in fs]
    outdir = './'
    listoftuples = [(f, outdir, ff) for f,ff in zip(fs,fits)]
    with multiprocessing.Pool(4) as pool:
        pool.starmap(alma_var.parallel_ms_in, listoftuples)
    """
    av = AlmaVar(ms_in=ms_in, clean_comp_fits=fits_in, outdir=outdir, log_level='INFO')
    av.process()


def parallel_ms_avg(ms_avg):
    """Helper to run multiprocessing, use with pool.starmap.

    Run with commands like:
    import alma_var
    import glob
    import multiprocessing
    fs = glob.glob('/path/to/ms*')
    with multiprocessing.Pool() as pool:
        pool.map(alma_var.parallel, fs)
    """
    av = AlmaVar(ms_avg=ms_avg, log_level='INFO')
    av.process()


def export_ms(msfilename, xcor=True, acor=False, reweight=True):
    """Direct copy of Luca Matra's export.
    https://github.com/dlmatra/miao
    """

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

    # Extract real and imaginary part of the visibilities at all u-v coordinates,
    # for both polarization states (XX and YY), extract weights which correspond
    # to 1/(uncertainty)^2
    Re_xx = data[0,:,:].real
    Im_xx = data[0,:,:].imag
    weight_xx = weight[0,:]
    if npol>=2:
        Re_yy = data[1,:,:].real
        Im_yy = data[1,:,:].imag
        weight_yy = weight[1,:]
        weight_xx = np.tile(weight_xx, nchan).reshape(nchan, -1)
        weight_yy = np.tile(weight_yy, nchan).reshape(nchan, -1)

        # Since we don't care about polarization, combine polarization states
        # (average them together) and fix the weights accordingly. Also if any
        # of the two polarization states is flagged, flag the outcome of the
        # combination.
        flags = np.logical_or(flags[0,:,:],flags[1,:,:])
        Re = np.where((weight_xx + weight_yy) != 0, (Re_xx*weight_xx + Re_yy*weight_yy) / (weight_xx + weight_yy), 0.)
        Im = np.where((weight_xx + weight_yy) != 0, (Im_xx*weight_xx + Im_yy*weight_yy) / (weight_xx + weight_yy), 0.)
        wgts = (weight_xx + weight_yy)
    else:
        Re=Re_xx
        Im=Im_xx
        wgts=weight_xx
        flags=flags[0,:,:]

    # Find which of the data represents cross-correlation between two
    # antennas as opposed to auto-correlation of a single antenna.
    # We don't care about the latter so we don't want it.
    xc = np.where(ant1 != ant2)[0]
    ac = np.where(ant1 == ant2)[0]
    if xcor:
        if acor:
            xc = np.logical_or(xc, ac)
    elif acor:
        xc = ac

    # Select data
    time = time[xc]
    spws = spwid[xc]
    # scan = scan[xc]
    data_real = Re[:, xc]
    data_imag = Im[:, xc]
    flags = flags[:, xc]
    data_wgts = wgts[:, xc]
    data_uu = uu[:, xc]
    data_vv = vv[:, xc]
    time = np.tile(time, data_uu.shape[0]).reshape(data_uu.shape[0], -1)
    spws = np.tile(spws, data_uu.shape[0]).reshape(data_uu.shape[0], -1)

    # Select only data that is NOT flagged, this step has the unexpected
    # effect of flattening the arrays to 1d
    data_real = data_real[np.logical_not(flags)]
    data_imag = data_imag[np.logical_not(flags)]
    # flags_ = flags[np.logical_not(flags)]
    data_wgts = data_wgts[np.logical_not(flags)]
    data_uu = data_uu[np.logical_not(flags)]
    data_vv = data_vv[np.logical_not(flags)]
    time = time[np.logical_not(flags)]
    spws = spws[np.logical_not(flags)]
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
    spws = spws[srt]

    # re-weighting as suggested by Loomis+
    # this assumes 1 d.o.f per visibility
    if reweight:
        spw_u = np.unique(spws)
        for s in spw_u:
            ok = spws == s
            wgt_mean = np.mean(data_wgts[ok])
            data_std = np.std(vis[ok])
            rew = (1/data_std**2)/wgt_mean
            logging.info(f're-weighting spw {s}: mean:{wgt_mean}, std:{data_std}')
            logging.info(f're-weighting spw {s} value (1dof): {rew}')
            data_wgts[ok] *= rew

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
        Model visibilities, 1d [nvis x nptsrc] or 2d [nvis, nptsrc]
    wt : numpy array
        Data weights, 1d [nvis]
    """

    #     R_inv = np.identity(len(wt))
    norm = 1/np.sqrt(np.sum(wt))
    if vis.shape == wt.shape:
        #         vis = vis_[:,np.newaxis]
        #         norm = 1 / np.sqrt( np.matmul(vis.conj().T, np.matmul(wt*R_inv, vis)) ).squeeze()
        #         h = norm * np.matmul(wt*R_inv, vis).squeeze()
        h = norm * wt * vis
    else:
        #         norm = 1 / np.sqrt(np.einsum('ij,ij->j', vis_.conj(), np.matmul(wt*R_inv, vis_)) )
        #         h = norm * np.matmul(wt*R_inv, vis_)
        h = norm * wt[:, np.newaxis] * vis

    return h, norm


def ptsrc_vis(u, v, ra, dec, flatxy=False):
    """Return visibilities of a point source at some ra/dec offset.

    Most of the computation time is in the exponential, and
    fiddling the outer product doesn't help much.

    Parameters
    ----------
    u, v : numpy array
        Points in u,v space (nvis)
    ra,dec : numpy array
        Offsets of points from phase center (1 or 2d)
    flatxy : bool, optional
        Return visibilities as flattened array [nuv x npt],
        rather than 2d [nuv, npt].
    """
    out = np.outer(u, -ra) + np.outer(v, dec)
    arg = 2*np.pi*1j*out
    if NUMEXPR:
        a = numexpr.evaluate('exp(arg)')
    else:
        a = np.exp(arg)

    if flatxy:
        return a.reshape((-1,))
    else:
        return a.reshape((-1,)+ra.shape)


def mjd2date(d):
    """Return a datetime object given an MJD."""
    t0 = datetime.datetime(1, 1, 1, 12)
    dt = datetime.timedelta(2400000.5 + d - 1721426.0)
    return t0 + dt


def plot_fits_sources(fits, ra, dec):
    """Plot sources on a FITS image.

    Multiple hacks needed...

    Parameters
    ----------
    fits: str
        Path to FITS file to plot.
    ra,dec : list
        Lists of RA and Dec of objects to include on plot.
    """

    # fix for UTC in fits file, but astropy wants utc
    h = astropy.io.fits.open(fits)
    # hack for UTC in file but astropy wants utc
    h[0].header['TIMESYS'] = h[0].header['TIMESYS'].lower()

    fig = aplpy.FITSFigure(h[0])
    fig.show_colorscale(stretch='linear', cmap='viridis')
    if len(ra) > 0:
        s = SkyCoord(ra, dec)
        fig.show_markers(s.ra, s.dec, marker='o', edgecolor='lightgrey')
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
    fig.tick_labels.set_xformat('hh:mm:ss')
    fig.tick_labels.set_yformat('dd:mm:ss')
    fig.set_title(fits.replace('.fits', ''))
    fig.savefig(fits.replace('.fits', '.png'))
    fig.close()
    h.close()


def get_ms_info(msfilepath):
    """Return various pieces of info from an ms.

    Primary beam, Technical Handbook S10.3.1

    Resolution approximation from:
    https://almascience.eso.org/about-alma/alma-basics

    Parameters
    ----------
    msfilepath : str
        Path to .ms file.
    """
    tb.open(msfilepath)
    uvw = tb.getcol("UVW")
    obsids = np.unique(tb.getcol('OBSERVATION_ID'))
    times = tb.getcol("TIME") / (24*60*60)
    tb.close()

    msmd.open(msfilepath)
    diams = msmd.antennadiameter()
    diam_unit = un.Unit(diams['0']['unit'])
    diams = [diams[k]['value'] for k in diams.keys()]
    unique_diams = np.unique(diams)
    diams = diams * diam_unit
    if len(unique_diams) > 1:
        logging.warning(f'more than one antenna diameter in {msfilepath}: {unique_diams}')
    spws = np.array([], dtype=int)
    for obsid in obsids:
        tmp = msmd.spwsforscans(obsid=obsid)
        for k in tmp.keys():
            spws = np.append(spws, tmp[k])
    spws = np.unique(np.array(spws))

    diam = np.mean(diams)
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

    field_id = np.unique([f for s in spws for f in msmd.fieldsforspw(s)])
    if len(field_id) > 1:
        logging.warning(f'{len(field_id)} fields for spws {spws}')
    field_id = field_id[0]
    field_name = msmd.namesforfields(field_id)[0]
    intent = np.unique(msmd.intentsforfield(field_id))

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
            'field_name': field_name,
            'diams': unique_diams,
            'intent': intent}

    return info


def clean_image(ms, datacolumn, outpath=None,
                niter=50000, cycleniter=500, nsigma=2.5,
                oversample=3, pb_factor=1.6,
                overwrite=False, tmpimage=None,
                subtract=False):
    """Make a clean image from an ms file.

    Parameters
    ----------
    ms : str
        Path to ms file.
    datacolumn : str
        Datacolumn from ms to use for clean.
    outpath : str, optional
        Path of output FITS file, default is ms path of ms.
    niter : int, optional
        Number of total clean iterations.
    cycleniter : int, optional
        Number of iterations per clean cycle.
    nsigma : float, optional
        N sigma at which to automatically stop cleaning.
    oversample : int, optional
        Factor to oversample PSF in images (pixel size = res / 2 / oversample).
    pb_factor : float, optional
        Number of primary beam FWHM for target search/clean images.
        1.6 gives images out to tclean default of pblimit=0.2.
    overwrite : bool, optional
        Overwrite existing FITS files.
    tmpimage : str, optional
        Path to temporary images while cleaning.
    subtract : bool, optional
        Subtract a model of the image and put in the CORRECTED column.
    """

    info = get_ms_info(ms)
    if outpath is None:
        outpath = os.path.dirname(ms)

    outpath = os.path.expanduser(outpath.rstrip('/'))

    if tmpimage is None:
        tmpimage = f'/tmp/tmpimage{str(randint(100000))}'

    # get a sensible looking image. pblimit is by default 0.2, but fwhm is 0.5,
    sizes = np.array([256, 320, 360, 384, 480, 500, 512, 1024, 2048])
    res_arcsec = info['spatial_res'].to(un.arcsec).value
    pix_sz = res_arcsec / 2 / oversample
    res_pix = res_arcsec / pix_sz
    img_fov = info['pb_hwhm'].to(un.arcsec).value * 2 * pb_factor
    img_sz = int(img_fov / pix_sz)
    img_sz = sizes[np.argmin(np.abs(img_sz - sizes))]
    pix_sz = img_fov / img_sz

    # figure out fields
    msmd.open(ms)
    scans = msmd.scannumbers()
    fields = msmd.fieldsforscans(scans)
    names = msmd.namesforfields(fields)
    msmd.close()

    # loop over fields and do ft for each
    for name in names:

        outfits = f'{outpath}/{name}_{datacolumn}.fits'

        # return if any fits already made, avoid subtraction issues
        if not overwrite and os.path.exists(outfits):
            logging.info(f'image {outfits} exists')
            return

        logging.info(f'making image {outfits} with cell:{pix_sz}, size:{img_sz}')

        if overwrite and os.path.exists(outfits):
            os.unlink(outfits)

        tclean(vis=ms, imagename=tmpimage, field=name,
               cell=f'{pix_sz}arcsec', imsize=[img_sz, img_sz],
               niter=niter, cycleniter=cycleniter, nsigma=nsigma,
               gain=0.2, deconvolver='multiscale', scales=[0, int(res_pix), int(4*res_pix)],
               interactive=False,
               datacolumn=datacolumn)
        exportfits(imagename=f'{tmpimage}.image',
                   fitsimage=outfits)

        # subtract model from data
        if subtract:
            logging.info(f'model {name} subtracted from visibilities in {ms}')
            # clear CORRECTED column, it will be filled from DATA each time
            tb.open(ms, nomodify=False)
            if 'CORRECTED_DATA' in tb.colnames():
                tb.removecols("CORRECTED_DATA")
            tb.close()
            # ft uses model (Jy/pix), not image (Jy/beam)
            ft(vis=ms, model=f'{tmpimage}.model', field=name,
               usescratch=True, incremental=False)

        os.system(f'rm -rf {tmpimage}*')

    # now subtract all fields
    logging.info(f'all models subtracted from visibilities in {ms}')
    if subtract:
        uvsub(vis=ms)


def subtract_fits_model(ms, fits):
    """Subtract a FITS image model from an ms.

    Parameters
    ms : str
        Path to ms file.
    fits : str
        Path to FITS image.
    """
    # clear CORRECTED column, it will be filled from DATA each time
    tb.open(ms, nomodify=False)
    if 'CORRECTED_DATA' in tb.colnames():
        tb.removecols("CORRECTED_DATA")
    tb.close()

    # ft uses model (Jy/pix), not image (Jy/beam)
    # invent a beam size to supress warnings
    with astropy.io.fits.open(fits) as h:
        aspp = np.abs(h[0].header['CDELT1']) * 3600
    tmpimage = f'/tmp/tmpimage{str(randint(100000))}'
    importfits(fitsimage=fits, imagename=f'{tmpimage}',
               beam=[f'{aspp:.3f}arcsec', f'{aspp:.3f}arcsec', '0deg'])
    ft(vis=ms, model=f'{tmpimage}',
       usescratch=True, incremental=False)
    os.system(f'rm -rf {tmpimage}*')
    h.close()

    # now subtract
    uvsub(vis=ms)
    logging.info(f'model subtracted from visibilities in {ms}')


def get_gaia_offsets(ra, dec, radius, date, min_plx_mas=None):
    """Return offsets of Gaia sources from ra/dec in radians.

    Parameters
    ----------
    ra, dec : Quantity, angle
        Center for query
    radius : Quantity, angle
        Radius of query
    date : datetime
        Date of observation
    min_plx_mas : float
        Minimum parallax to apply to query result. Simply a
        way to cut down returned objects to nearby stars.

    Returns
    -------
    tuple
        RA/Dec offsets of targets in radians, and full result table.
    """
    coord = SkyCoord(ra=ra, dec=dec, frame='icrs')
    logging.info(f'get_gaia_offsets: search within {radius} at {ra}, {dec}')

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

    # convert to sky offsets w.r.t. query center
    ra_off = np.array((r['ra_ep'] - ra).to(un.rad).value)
    dec_off = np.array((r['dec_ep'] - dec).to(un.rad).value)
    ra_off = (ra_off + np.pi) % (2 * np.pi) - np.pi
    dec_off = (dec_off + np.pi) % (2 * np.pi) - np.pi
    # RA offset is ra x cos(dec)
    ra_off *= np.cos(dec.to(un.rad).value)

    return ra_off, dec_off, r


def tiled_offsets(radius, res):
    """Return tiled offset points

    Parameters
    ----------
    radius : astropy unit quantity
        Radius to tile out to.
    res : astropy unit quantity
        Spatial resolution.
    """
    n = 2*radius.to(un.rad).value / res.to(un.rad).value
    x = (np.arange(n) - n//2) * res.to(un.rad).value
    xx, yy = np.meshgrid(x, x)
    r = np.hypot(xx, yy)
    ok = r <= radius.to(un.rad).value
    return xx[ok], yy[ok]


def randint(n):
    seed = int.from_bytes(os.urandom(10), byteorder='big')
    return np.random.default_rng(seed).integers(n)


class AlmaVar:

    def __init__(self, ms_in=None, ms_avg=None, outdir=None,
                 clean_comp_fits=None,
                 det_snr=4, pb_factor=1.6, auto_load=False,
                 log_level='WARNING'):
        """Initialise by creating output folder and averaging ms.

        Parameters
        ----------
        ms_in : str
            Path of ms file to process. Not needed if ms_avg given.
        ms_avg : str
            Path of already averaged ms file if exists.
        outdir : str, optional
            Folder in which to put output (subfolder will be created here).
             Not needed if ms_avg given.
        det_snr : float, optional
            SNR threshold for flagging a detection.
        auto_load : bool, optional
            Automatically load what has been processed already.
            May cause problems if settings have changed.
        pb_factor : float, optional
            Number of primary beam FWHM for target search/clean images.
            1.6 gives images out to tclean default of pblimit=0.2.
        log_level : str, optional
            Level of logging, INFO or WARNING.
        """
        # set up some output folders
        if outdir:
            outdir = os.path.expanduser(outdir.rstrip('/'))
        if ms_in:
            ms_in = os.path.expanduser(ms_in.rstrip('/'))
            self.ms_in = os.path.abspath(ms_in)
            self.ms_in_name = os.path.basename(self.ms_in)
            if outdir is None:
                outdir = os.path.dirname(self.ms_in)
            self.wdir = f'{outdir}/{self.ms_in_name}.var'
            self.ms_avg = f'{self.wdir}/{self.ms_in_name}.avg'
        else:
            if '.avg' not in ms_avg:
                logging.warning(f'ms_avg should end in ".avg"')
            ms_avg = os.path.expanduser(ms_avg.rstrip('/'))
            self.ms_avg = os.path.abspath(ms_avg)
            self.wdir = os.path.dirname(f'{self.ms_avg}/../')

        self.scandir = f'{self.wdir}/scans'

        if not os.path.exists(self.wdir):
            os.mkdir(self.wdir)

        casalog.setlogfile(f'{self.wdir}/casa.log')
        casalog.filterMsg(['Restoring with an empty model image'])  # don't need this warning
        logging.basicConfig(filename=f'{self.wdir}/almavar.log',
                            format='%(levelname)s:%(message)s',
                            level=logging.getLevelName(log_level),
                            force=True)

        logging.info(f'output folder {self.wdir}')

        if not os.path.exists(self.scandir):
            os.mkdir(self.scandir)

        # only need info from ms_in if we are going to average it
        if os.path.exists(self.ms_avg):
            self.fill_ms_avg_info()
        else:
            ms.open(self.ms_in)
            self.ms_in_spw_info = ms.getspectralwindowinfo()
            ms.close()

            # check for datacolumn
            tb.open(self.ms_in)
            self.ms_in_datacol = 'CORRECTED'
            if 'CORRECTED_DATA' not in tb.colnames():
                self.ms_in_datacol = 'DATA'
            tb.close()

        # some things (that might get filled later)
        self.clean_comp_fits = clean_comp_fits
        self.pb_factor = pb_factor
        self.det_snr = det_snr
        self.field_sources = {}
        self.scan_info = None
        self.scan_vis = None

        # auto load will go here
        if auto_load:
            pass

    def process(self, subtract=True):
        """Shortcut."""
        self.avg_ms_in(subtract=subtract)
        self.split_scans(keep_scan_ms=True)
        self.clean_images()
        self.split_scans(keep_scan_ms=False)  # delete ms files
        self.diagnostics()
        self.summed_filter()
        self.field_matched_filter()
        self.plot_sources()
        self.timeseries_summary()
        logging.info(f'finished {self.wdir}')

    def avg_ms_in(self, nchan_spw=1, intent='OBSERVE_TARGET#ON_SOURCE',
                  clean=True, subtract=True,
                  tdm=True, fdm=True, sqld=False, chavg=False):
        """Average ms_in down to fewer channels per spw.

        The resulting ms has the averaged visibilities in DATA, and if
        the subtraction is done, a subtracted set in CORRECTED_DATA.

        Parameters
        ----------
        nchan_spw : int, optional
            Number of channels per spw to average down to.
        intent : str or list of str, optional
            Intents to extract from input ms file, e.g. use
            'OBSERVE_TARGET#ON_SOURCE' to skip calibration scans.
        tdm, fdm, sqld, chavg : bool, optional
            Indicate which ALMA spws to include.
        clean : bool, optional
            Create clean images
        subtract : bool, optional
            Subtract a model of the averaged ms. Requires clean=True.
        """

        # skips ms_in related stuff if ms_avg exists already
        if not os.path.exists(self.ms_avg):
            logging.info('averaging input ms')

            # details, so we can exclude most data
            # TDM/FDM corresponds to FULL_RES (not SQLD or CH_AVG)
            # but we allow for various choices anyway
            msmd.open(self.ms_in)
            spws = np.array([])
            if fdm:
                spws = np.append(spws, msmd.almaspws(fdm=True))
            if tdm:
                spws = np.append(spws, msmd.almaspws(tdm=True))
            if sqld:
                spws = np.append(spws, msmd.almaspws(sqld=True))
            if chavg:
                spws = np.append(spws, msmd.almaspws(chavg=True))
            logging.info(f'spws: {spws}')

            # check any intents exist
            if len(msmd.intents()) == 0:
                logging.warning(f'no intents, keeping all scans')
                intent = ''

            msmd.close()

            # average FULL_RES down to fewer channels per spw
            # output is in DATA column
            spw_list = []
            avg_list = []
            for k in self.ms_in_spw_info.keys():
                spwid = self.ms_in_spw_info[k]['SpectralWindowId']
                if spwid in spws:
                    spw_list.append(spwid)
                    avg_list.append(self.ms_in_spw_info[k]['NumChan']//nchan_spw)

            logging.info(f'keeping spws:{spw_list}, widths:{avg_list}')
            split(vis=self.ms_in, outputvis=self.ms_avg, keepflags=False,
                  spw=','.join([str(s) for s in spw_list]), width=avg_list,
                  datacolumn=self.ms_in_datacol, intent=intent)
        else:
            logging.info('loading averaged ms')

        # make images and subtract continuum model
        if clean:
            if self.clean_comp_fits:
                clean_image(self.ms_avg, 'data', niter=0)
                if subtract:
                    subtract_fits_model(self.ms_avg, self.clean_comp_fits)
                    clean_image(self.ms_avg, 'corrected', niter=0)
            else:
                clean_image(self.ms_avg, 'data', subtract=subtract)
                if subtract:
                    clean_image(f'{self.ms_avg}', 'corrected', niter=0)

        # if not subtracting, MODEL_DATA is blank and we copy
        # DATA to CORRECTED_DATA by running uvsub
        if not subtract:
            logging.info('no model subtraction, copying DATA to CORRECTED')
            tb.open(self.ms_avg, nomodify=False)
            tb.renamecol('DATA', 'CORRECTED_DATA')
            tb.close()
            clean_image(self.ms_avg, 'corrected', niter=0)

        self.fill_ms_avg_info()

    def fill_ms_avg_info(self):
        """Fill some info for averaged ms."""
        ms.open(self.ms_avg)
        self.ms_avg_scan_info = ms.getscansummary()
        ms.close()

        # get spws in averaged data
        self.ms_avg_spws = {}
        tb.open(self.ms_avg)
        self.ms_avg_spws['all'] = np.unique(tb.getcol('DATA_DESC_ID'))
        tb.close()

        msmd.open(self.ms_avg)
        self.ms_avg_spws['tfdm'] = msmd.almaspws(tdm=True, fdm=True)
        self.ms_avg_spws['sqld'] = msmd.almaspws(sqld=True)
        self.ms_avg_spws['chavg'] = msmd.almaspws(chavg=True)
        msmd.close()

    def split_scans(self, scans=None, datacolumn='CORRECTED',
                    keep_scan_ms=False, load_scan_vis=False):
        """Split scans from averaged ms.

        Parameters
        ----------
        scans : list, optional
            List of scans to image, default is all scans
        datacolumn : str, optional
            Data column to split out, CORRECTED has average image subtracted
        keep_scan_ms : bool, optional
            Keep scan ms file
        load_scan_vis : bool, optional
            Load scan visibilities into this object.
        """
        if scans:
            scans_sorted = scans
        else:
            scans_sorted = [int(s) for s in self.ms_avg_scan_info.keys()]

        scans_sorted.sort()
        self.scan_info = {}
        self.scan_vis = {}
        for scan_no in scans_sorted:

            scan_no_dir = f'{self.scandir}/{scan_no}'
            if not os.path.exists(scan_no_dir):
                os.mkdir(scan_no_dir)

            scan_str = f'scan_{int(scan_no):02d}'

            scan_avg_vis = f'{scan_no_dir}/{scan_str}-vis.npy'
            scan_avg_info = scan_avg_vis.replace('-vis', '-info')
            scan_avg_ms = f'{scan_no_dir}/{scan_str}.ms'

            if os.path.exists(scan_avg_vis) and not keep_scan_ms:
                logging.info(f'loading scan {scan_no} from {scan_avg_vis}')
                u, v, vis, wt, time = load_npy_vis(scan_avg_vis)
                if len(np.unique(time)) == 0:
                    logging.warning(f'no time points for scan {scan_no}, skipping')
                else:
                    self.scan_info[scan_no] = np.load(scan_avg_info, allow_pickle=True).item()
            else:
                logging.info(f'splitting scan {scan_no} from {scan_avg_ms} using {datacolumn}')

                if not os.path.exists(scan_avg_ms):
                    split(vis=self.ms_avg, outputvis=scan_avg_ms,
                          scan=scan_no, datacolumn=datacolumn, keepflags=False)

                # get data from ms
                u, v, vis, wt, time = export_ms(scan_avg_ms)
                # save, this will make everything complex
                np.save(scan_avg_vis, np.array([u, v, vis, wt, time]))

                info = get_ms_info(scan_avg_ms)
                info['nvis'] = len(vis)
                info['scan_str'] = scan_str
                info['scan_dir'] = scan_no_dir
                info['scan_avg_ms'] = scan_avg_ms
                info['scan_avg_vis'] = scan_avg_vis
                if len(info['diams']) > 1:
                    logging.warning(f'scan {scan_no} has antenna diams {info["diams"]}')
                np.save(scan_avg_info, info)

                if len(np.unique(time)) == 0:
                    logging.warning(f'no time points for scan {scan_no}, skipping')
                else:
                    self.scan_info[scan_no] = info

            if load_scan_vis:
                self.scan_vis[scan_str] = {'u': u, 'v': v, 'vis': vis, 'wt': wt, 'time': time}

            if len(vis) == 0:
                logging.info(f'{scan_no}: no visibilities')

            if not keep_scan_ms and os.path.exists(scan_avg_ms):
                shutil.rmtree(scan_avg_ms)

    def diagnostics(self, scans=None, reloutdir='.',
                    pcrit=0.001):
        """Various sanity checks and diagnostic plots.

        Parameters
        ----------
        scans : list, optional
            List of scans to image, default is all scans
        reloutdir : str
            Relative location from savefile in which to put output plots.
        pcrit : float, optional
            Critical p value below which to flag data as non-normal.
            Values for calibration scans tend to be <0.0005 or so.
        """
        if scans is None:
            scans = [s for s in self.scan_info.keys()]

        for scan in scans:
            logging.info(f'sanity checking scan {scan}')
            u, v, vis, wt, time = load_npy_vis(self.scan_info[scan]['scan_avg_vis'])
            if len(vis) == 0:
                continue

            outpath = f"{os.path.dirname(self.scan_info[scan]['scan_avg_vis'])}/{reloutdir}"
            if not os.path.exists(outpath):
                os.mkdir(outpath)

            # check normality of distribution at each timestep
            times = np.unique(time)
            self.scan_info[scan]['flagged_times'] = np.array([])
            for i, t in enumerate(times):
                ok = time == t
                if np.sum(ok) > 3:
                    res = scipy.stats.shapiro(vis.real[ok]*np.sqrt(wt[ok]))
                    res_i = scipy.stats.shapiro(vis.imag[ok]*np.sqrt(wt[ok]))
                    if res_i.pvalue < res.pvalue:
                        res = res_i
                    pvalue = res.pvalue
                else:
                    pvalue = pcrit
                if pvalue <= pcrit:
                    self.scan_info[scan]['flagged_times'] = np.append(self.scan_info[scan]['flagged_times'], t)
                    logging.warning(f'distribution non-normal with p {pvalue:.4f}%')
                    logging.warning(f' in scan {scan} at time {t} (step {i}) with'
                                    f" intent {self.scan_info[scan]['intent']}")

            # check visibility weights sensible
            # multiply by sqrt(2) assuming Re and Im independent
            fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
            dr = np.sqrt(2)*vis.real*np.sqrt(wt)
            di = np.sqrt(2)*vis.imag*np.sqrt(wt)
            mx = 0
            for a, d, l in zip(ax, [dr, di], ['Real', 'Imag']):
                x = np.linspace(np.min(d), np.max(d), 100)
                _ = a.hist(d, bins=100, log=True, label=l)
                a.plot(x, np.max(_[0])*np.exp(-((x-np.mean(d))**2)/2))
                a.set_xlabel('snr per visibility')
                a.legend()
                if np.max(_[0]) > mx:
                    mx = np.max(_[0])
            ax[0].set_ylim(0.5, 2*mx)
            ax[0].set_ylabel('number')
            fig.tight_layout()
            fig.savefig(f"{outpath}/{self.scan_info[scan]['scan_str']}_vis_snr.png")
            plt.close(fig)

            # look at raw and summed data
            tplot1 = (time-np.min(time))*24*60
            fig, ax = plt.subplots(3, sharex=True, figsize=(8, 6))
            ax[0].plot(tplot1, vis.real, '.', label='Real', markersize=0.3)
            ax[1].plot(tplot1, vis.imag, '.', label='Imag', markersize=0.3)
            ax[2].plot(tplot1, wt, '.', label='Weight', markersize=0.3)
            ax[2].set_xlabel('Time / minutes')
            ax[0].set_ylabel('Real')
            ax[1].set_ylabel('Imag')
            ax[2].set_ylabel('Weight')
            fig.tight_layout()
            fig.savefig(f"{outpath}/{self.scan_info[scan]['scan_str']}_rawvis_time.png")
            plt.close(fig)

    def clean_images(self, scans=None, overwrite=False):
        """Create tclean images for specified scans.

        Parameters
        ----------
        scans : list, optional
            List of scans to image, default is all scans
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

            clean_image(self.scan_info[scan]['scan_avg_ms'], 'data',
                        outpath=self.scan_info[scan]['scan_dir'], overwrite=overwrite, niter=0)

    def link_to_field(self, file, scan):
        """Symlink a file in a field directory."""
        fn = self.scan_info[scan]['field_name']

        fdir = f'{self.wdir}/{fn}'
        if not os.path.exists(fdir):
            os.mkdir(fdir)

        link = f'{fdir}/{os.path.basename(file)}'
        link_rel = os.path.relpath(os.path.dirname(file), fdir)
        linked = f'{link_rel}/{os.path.basename(file)}'
        if not os.path.exists(link):
            os.symlink(linked, link)

    def summed_filter(self, scans=None, plot_vis_hist=False, plot_raw_vis=False):
        """Summed search for specified scans."""
        if scans is None:
            scans = [s for s in self.scan_info.keys()]

        for scan in scans:
            if self.scan_info[scan]['nvis'] > 0:
                det, fn = self.summed_search(scan, self.det_snr,
                                             outpre=self.scan_info[scan]['scan_str'],
                                             plot_vis_hist=plot_vis_hist,
                                             plot_raw_vis=plot_raw_vis)
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
                dets, fns = self.matchedfilter_search(scan, self.det_snr,
                                                      outpre=self.scan_info[scan]['scan_str'],
                                                      ra_off=ra_off, dec_off=dec_off)
                for det, fn in zip(dets, fns):
                    if det:
                        self.link_to_field(fn, scan)

    def init_field_sources(self, update=False):
        scans = [s for s in self.scan_info.keys()]
        for scan in scans:
            field = self.scan_info[scan]['field_id']
            if field not in self.field_sources.keys() or update:
                self.field_sources[field] = {}
                self.field_sources[field]['scan'] = scan  # all scans of field have same pointing
                self.field_sources[field]['sources'] = []
                self.field_sources[field]['ra_off'] = []
                self.field_sources[field]['dec_off'] = []

    def gaia_sources(self, min_plx_mas=None, update=False):
        """Find and save Gaia DR3 sources in FOV for specified scans.

        Sources are specified per field, which might be observed
        multiple times, so more efficient to save per field.
        """
        self.init_field_sources(update=update)

        for field in self.field_sources.keys():
            if 'gaia' not in self.field_sources[field]['sources']:
                ra_off, dec_off, _ = get_gaia_offsets(self.scan_info[self.field_sources[field]['scan']]['phase_center'][0],
                                                      self.scan_info[self.field_sources[field]['scan']]['phase_center'][1],
                                                      self.scan_info[self.field_sources[field]['scan']]['pb_hwhm'] * self.pb_factor,
                                                      self.scan_info[self.field_sources[field]['scan']]['mean_time'],
                                                      min_plx_mas=min_plx_mas)
                self.field_sources[field]['sources'].append('gaia')
                self.field_sources[field]['ra_off'].append(ra_off)
                self.field_sources[field]['dec_off'].append(dec_off)
                logging.info(f'found {len(ra_off)} gaia sources for field {field}')

    def plot_sources(self, scans=None):
        """Plot sources on a FITS file if it exists."""
        if scans is None:
            scans = [s for s in self.scan_info.keys()]

        for scan in scans:
            field = self.scan_info[scan]['field_id']

            fits = (f"{self.scan_info[scan]['scan_dir']}/"
                    f"{self.scan_info[scan]['field_name']}_data.fits")

            if os.path.exists(fits):
                plot_fits_sources(fits,
                                  self.field_sources[field]['ra_off']*un.rad + self.scan_info[scan]['phase_center'][0],
                                  self.field_sources[field]['dec_off']*un.rad + self.scan_info[scan]['phase_center'][1])

    def field_matched_filter(self, scans=None, min_plx_mas=None):
        """Run matched filter for Gaia sources in FOV."""
        if scans is None:
            scans = [s for s in self.scan_info.keys()]

        for scan in scans:
            if self.scan_info[scan]['nvis'] > 0:
                field = self.scan_info[scan]['field_id']

                if field not in self.field_sources.keys():
                    self.gaia_sources(min_plx_mas=min_plx_mas)

                if len(self.field_sources[field]['ra_off']) == 0:
                    continue

                self.matched_filter(scans=[scan],
                                    ra_off=self.field_sources[field]['ra_off'],
                                    dec_off=self.field_sources[field]['dec_off'])

    def smooth_plot(self, times, v_in, det_snr, scan,
                    show_sig=True, ylab='$|Sum(V.w)|$',
                    outdir='.', outpre='', outfile=f'vis_time_smooth.png'):
        """Diagnostic plot for time series, return True for detection."""
        if len(times) < 1:
            return False, None

        t0 = np.min(times)
        tplot2 = (times-t0)*24*60
        dt = np.median(np.diff(times))*24*60*60  # dt in seconds
        nw = 60
        if len(times)/2 < nw:
            nw = len(times)//2
        if nw == 0:
            nw = 1
        ws = np.arange(nw)+1
        T = np.zeros((nw, len(times)))
        wpk = []
        pk = []
        snr = []

        # find outliers, empirically or assuming SNR is properly distributed
        vmin = 1e5
        for i, wi in enumerate(ws):
            conv = np.convolve(v_in, np.repeat(1, wi)/wi, mode='valid')

            if ylab == 'SNR':
                T[i, (wi-1)//2:len(v_in)-wi//2] = conv * np.sqrt(wi)

            else:
                if len(times) > 1:
                    clipped, _, _ = scipy.stats.sigmaclip(conv, low=3, high=3)
                    mn, std = np.mean(clipped), np.std(clipped)
                    conv = (conv-mn) / std
                else:
                    conv = 1
                T[i, (wi-1)//2:len(v_in)-wi//2] = conv

            ok = np.where(T[i] > det_snr)[0]
            for o in ok:
                wpk.append(wi)
                pk.append(tplot2[o])
                snr.append(T[i, o])

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
        if len(self.scan_info[scan]['flagged_times']) > 0:
            fplot = (self.scan_info[scan]['flagged_times']-np.min(times))*24*60
            ax[1].vlines(fplot, np.min(v_in), np.max(v_in),
                         linewidth=5, alpha=0.2, color='grey',
                         label='suspect')
            ax[1].legend()
        ax[1].plot(tplot2, v_in, '.', label=ylab)
        t0_str = astropy.time.Time(t0, format='mjd').iso
        ax[1].set_xlabel(f'Time - {t0_str} / minutes')
        ax[1].set_ylabel(ylab)
        ax[0].set_ylabel(f'smoothing width / $\Delta t$={dt:3.2f} seconds')
        fig.tight_layout()
        figname = f'{outdir}/{outpre}{outfile}'
        fig.savefig(figname)
        plt.close(fig)

        return len(wpk) > 0, figname

    def summed_search(self, scan, det_snr, reloutdir='.', outpre='',
                      plot_vis_hist=False, plot_raw_vis=False):
        """Point source position-free search.

        Parameters
        ----------
        scan : str
            Scan to run on.
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
        savefile = self.scan_info[scan]['scan_avg_vis']
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
            fig.savefig(f'{os.path.dirname(savefile)}/{reloutdir}/{outpre}_vis_snr.png')
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
            fig.savefig(f'{os.path.dirname(savefile)}/{reloutdir}/{outpre}_rawvis_time.png')
            plt.close(fig)

        # smooth light curve and plot
        outpath = f'{os.path.dirname(savefile)}/{reloutdir}'
        if not os.path.exists(outpath):
            os.mkdir(outpath)

        return self.smooth_plot(times, v_abs, det_snr, scan,
                                outdir=f'{outpath}', outfile=f'{outpre}_sum.png')

    def matchedfilter_search(self, scan, det_snr, ra_off=None, dec_off=None,
                             save=True, reloutdir='matchf', outpre=''):
        """Run matched filter search on saved set of visibilities.

        Search runs over multiple sources at each time step.
        vis_mod is the model and h is the matched filter, at each
        time both have shape [nvis, npt]. The resulting array v_pos
        has shape [ntime, npt].

        Parameters
        ----------
        scan : str
            Scan to run on.
        det_snr : float
            SNR threshold for detection flagging.
        ra_off, dec_off : numpy array, list, tuple
            Coordinates for search in radians
        save : bool
            Save time, snr, flux arrays.
        reloutdir : str
            Relative location from savefile in which to put output plots.
        outpre : str, optional
            String to prepend to filename.
        """
        savefile = self.scan_info[scan]['scan_avg_vis']
        logging.info(f'matched filter: loading from {savefile}')
        u, v, vis, wt, time = load_npy_vis(savefile)

        if ra_off is None:
            ra_off = [0.]
            dec_off = [0.]

        # get ra/dec into 1d arrays of positions
        ra = np.array(ra_off).flatten()
        dec = np.array(dec_off).flatten()

        times = np.unique(time)

        snr = []
        flux = []

        for t in times:

            ok = time == t
            vis_mod = ptsrc_vis(u[ok], v[ok], ra, dec)
            h, norm = h_filter(vis_mod, wt[ok])
            snr_ = np.sqrt(2) * np.real(np.dot(vis[ok], h.conj()))
            snr.append(snr_)
            flux.append(snr_ * norm)

        snr = np.array(snr)
        flux = np.array(flux)
        if save:
            np.savez(self.scan_info[scan]['scan_avg_vis'].replace('-vis.npy', '-time_snr_flux.npz'),
                     time=times, flux=flux.T, snr=snr.T, ra=np.rad2deg(ra)*3600, dec=np.rad2deg(dec)*3600)

        # plots
        outpath = f'{os.path.dirname(savefile)}/{reloutdir}'
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        dets = []
        fns = []
        for i in range(len(ra)):
            det, fn = self.smooth_plot(times, snr[:, i], det_snr, scan,
                                       outdir=outpath, outfile='.png', ylab='SNR',
                                       outpre=(f'{outpre}_{i:03d}_{np.rad2deg(ra[i])*3600:.3f}'
                                               f'_{np.rad2deg(dec[i])*3600:.3f}'))
            dets.append(det)
            fns.append(fn)

        return dets, fns

    def uvmodelfit_search(self, msfilepath, ra_off=None, dec_off=None, dt=None,
                          reloutdir='uvmfit', make_fits=False, vary_pos=False,
                          cleanpar=None):
        """Run uvmodelfit search on a single-scan ms.

        Parameters
        ----------
        msfilepath : str
            Path to ms file.
        ra_off, dec_off : numpy array, list, tuple
            Coordinates for search in radians.
        dt : float, optional
            Time step, defaults to individual integration time.
        reloutdir : str, optional
            Relative location from savefile in which to put output plots.
        make_fits : bool, optional
            Make a series of FITS files with images
        vary_pos : bool
            Allow position to vary in fit.
        cleanpar : dict, optional
            Dict of args for tclean, need cell and imsize.
        """

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
            make_fits_ = make_fits
            for ra, dec in zip(ra_off, dec_off):
                self.uvmodelfit_search(msfilepath, ra, dec, dt=dt, reloutdir=reloutdir,
                                       cleanpar=cleanpar, make_fits=make_fits_, vary_pos=vary_pos)
                make_fits_ = False  # only first time around
            return

        # convert offsets to arcsec
        ra_off = np.rad2deg(ra_off) * 3600
        dec_off = np.rad2deg(dec_off) * 3600

        outdir = f"{os.path.dirname(msfilepath)}/{reloutdir.rstrip('/')}"
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        wdir = outdir
        if not os.path.exists(wdir):
            os.mkdir(wdir)
        cl = f'/tmp/{str(randint(100000))}.cl'
        if os.path.exists(cl):
            shutil.rmtree(cl)

        coords = f'{ra_off:.3f}_{dec_off:.3f}'

        logging.info(f'running uv search for {os.path.basename(msfilepath)}')

        flux = []
        ra = []
        dec = []

        tb.open(msfilepath)
        times = tb.getcol("TIME") / (24*60*60)
        tb.close()
        time = np.unique(times)
        dt = np.mean(np.diff(time))

        cleandir = f'{wdir}/cleans'
        if not os.path.exists(cleandir) and make_fits:
            os.mkdir(cleandir)

        for i, t in enumerate(time):

            t0_ = mjd2date(t-dt/2)
            t1_ = mjd2date(t+dt/2)
            t0_str = t0_.strftime('%H:%M:%S') + t0_.strftime(".%f")[:2]
            t1_str = t1_.strftime('%H:%M:%S') + t1_.strftime(".%f")[:2]

            # uvmodelfit, flux is saved in cl as complex
            uvmodelfit(vis=msfilepath, timerange=f"{t0_str}~{t1_str}",
                       comptype='P',
                       sourcepar=[1, ra_off, dec_off], varypar=[True, vary_pos, vary_pos],
                       outfile=cl)
            tb.open(cl)
            flux.append(np.real(tb.getcol('Flux')[0][0]))
            ra.append(np.real(tb.getcol('Reference_Direction')[0][0]))
            dec.append(np.real(tb.getcol('Reference_Direction')[1][0]))
            tb.close()

            if make_fits:
                if cleanpar is None:
                    cleanpar = {'cell': '0.5arcsec', 'imsize': [256, 256]}

                tmpimage = f'/tmp/tmpimage{str(randint(100000))}'
                tclean(vis=f'{msfilepath}', imagename=tmpimage,
                       **cleanpar, interactive=False, niter=0,
                       timerange=f"{t0_str}~{t1_str}")
                exportfits(imagename=f'{tmpimage}.image',
                           fitsimage=f'{cleandir}/{i:04d}.fits')
                os.system(f'rm -rf {tmpimage}*')

        t0 = np.min(time)
        tplot = (time-t0)*24*60

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tplot, flux, '.')
        t0_str = astropy.time.Time(t0, format='mjd').iso
        ax.set_xlabel(f'Time - {t0_str} / minutes')
        ax.set_ylabel('Flux / Jy')
        fig.savefig(f'{outdir}/{coords}_flux_time.png')
        plt.close(fig)
        np.save(f'{outdir}/{coords}_timeflux.npy', np.vstack((time, flux, ra, dec)))

        shutil.rmtree(cl)

    def timeseries_summary(self, plot=True):
        """Get and plot timeseries fluxes."""
        data = {}
        t0 = 1e6
        for s in self.scan_info.keys():
            file = self.scan_info[s]['scan_avg_vis'].replace('-vis.npy', '-time_snr_flux.npz')
            if os.path.exists(file):
                fh = np.load(file)
                time, flux, snr, ra, dec = fh['time'], fh['flux'], fh['snr'], fh['ra'], fh['dec']
                if np.min(time) < t0:
                    t0 = np.min(time)
                for i in range(len(ra)):
                    targ = str(ra[i])+','+str(dec[i])
                    if targ in data.keys():
                        data[targ] = {'coord': data[targ]['coord'],
                                      'time': np.append(data[targ]['time'], time),
                                      'flux': np.append(data[targ]['flux'], flux[i]),
                                      'snr': np.append(data[targ]['snr'], snr[i])}
                    else:
                        data[targ] = {'coord': f'{ra[i]:.3f},{dec[i]:.3f}',
                                      'time': time,
                                      'flux': flux[i],
                                      'snr': snr[i]}

        if plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, k in enumerate(data.keys()):
                off = 8 * np.std(data[k]['flux'])
                ax.text(np.min(data[k]['time']-t0), np.median(data[k]['flux'])+i*off, data[k]['coord'],
                        alpha=0.5, horizontalalignment='left')
                ax.plot(data[k]['time']-t0,
                        np.ones(len(data[k]['time']))*np.median(data[k]['flux'])+i*off, alpha=0.5)
                ax.scatter(data[k]['time']-t0, data[k]['flux'] + i*off, s=1)
            t0_str = astropy.time.Time(t0, format='mjd').iso
            ax.set_xlabel(f'time - {t0_str} / days')
            ax.set_ylabel('flux + offset / Jy')
            fig.tight_layout()
            fig.savefig(f'{self.wdir}/timeseries.pdf')

        return data
