import os
import shutil
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def mjd2date(d):
    t0 = datetime.datetime(1, 1, 1, 12)
    dt = datetime.timedelta(2400000.5 + d - 1721426.0)
    return t0 + dt


def export_ms(msfilename, tb, ms):
    '''Return visibilities etc.

    Direct copy of Luca Matra's export.'''

    cc=2.9979e10 #cm/s

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
    scan    = tb.getcol("SCAN_NUMBER")
    tb.close()
    
    if np.any(flags):
        logging.warning(f"{msfilename}: some of the data is FLAGGED")
    
    logging.info("Found data with "+str(data.shape[-1])+" uv points")

    ms.open(msfilename)
    spw_info = ms.getspectralwindowinfo()
    nchan = spw_info["0"]["NumChan"]
    npol = spw_info["0"]["NumCorr"]
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

    logging.info("Datasets has baselines between "+str(np.min(np.sqrt(uvw[0,:]**2.0+uvw[1,:]**2.0)))+" and "+str(np.max(np.sqrt(uvw[0,:]**2.0+uvw[1,:]**2.0)))+" m")

    #Initialize u and v arrays (coordinates in Fourier space)
    uu=np.zeros((freqs.shape[0],uvw[0,:].size))
    vv=np.zeros((freqs.shape[0],uvw[0,:].size))

    #Fill u and v arrays appropriately from data values.
    for i in np.arange(freqs.shape[0]):
        for j in np.arange(uvw.shape[1]):
            uu[i,j]=uvw[0,j]*freqs[i,spwid[j]]/(cc/100.0)
            vv[i,j]=uvw[1,j]*freqs[i,spwid[j]]/(cc/100.0)

    # Extract real and imaginary part of the visibilities at all u-v
    # coordinates, for both polarization states (XX and YY), extract
    # weights which correspond to 1/(uncertainty)^2
    Re_xx = data[0,:,:].real
    Re_yy = data[1,:,:].real
    Im_xx = data[0,:,:].imag
    Im_yy = data[1,:,:].imag
    weight_xx = weight[0,:]
    weight_yy = weight[1,:]

    # Since we don't care about polarization, combine polarization states
    # (average them together) and fix the weights accordingly. Also if
    # any of the two polarization states is flagged, flag the outcome of
    # the combination.
    flags = flags[0,:,:]*flags[1,:,:]
    Re = np.where((weight_xx + weight_yy) != 0, (Re_xx*weight_xx + Re_yy*weight_yy) / (weight_xx + weight_yy), 0.)
    Im = np.where((weight_xx + weight_yy) != 0, (Im_xx*weight_xx + Im_yy*weight_yy) / (weight_xx + weight_yy), 0.)
    wgts = (weight_xx + weight_yy)

    # Find which of the data represents cross-correlation between two
    # antennas as opposed to auto-correlation of a single antenna.
    # We don't care about the latter so we don't want it.
    xc = np.where(ant1 != ant2)[0]

    # Select only cross-correlation data
    time = time[xc]
    scan = scan[xc]
    data_real = Re[:,xc]
    data_imag = Im[:,xc]
    flags = flags[:,xc]
    data_wgts = wgts[xc]
    data_uu = uu[:,xc]
    data_vv = vv[:,xc]
    data_wgts=np.reshape(np.repeat(wgts[xc], uu.shape[0]), data_uu.shape)

    # Select only data that is NOT flagged, this step has the unexpected
    # effect of flattening the arrays to 1d
    data_real = data_real[np.logical_not(flags)]
    data_imag = data_imag[np.logical_not(flags)]
    flagss = flags[np.logical_not(flags)]
    data_wgts = data_wgts[np.logical_not(flags)]
    data_uu = data_uu[np.logical_not(flags)]
    data_vv = data_vv[np.logical_not(flags)]
    time = time[np.logical_not(flags[0])]
    scan = scan[np.logical_not(flags[0])]

    time /= (24*60*60) # to MJD
    vis = data_real + 1j*data_imag
    
    # sort into time order
    srt = np.argsort(time)
    data_uu = data_uu[srt]
    data_vv = data_vv[srt]
    vis = vis[srt]
    data_wgts = data_wgts[srt]
    time = time[srt]

    return data_uu, data_vv, vis, data_wgts, time
    

def var_search(msfile, keep_avg_ms=True, keep_scan_ms=False, keep_scan_npy=True):

    logging.info(f'running search for {msfile}')

    wdir = f'{msfile}.var'
    if not os.path.exists(wdir):
        os.mkdir(wdir)
        
    scandir = f'{wdir}/scans'
    if not os.path.exists(scandir):
        os.mkdir(scandir)

    # get spw info
    ms.open(msfile)
    spw_info = ms.getspectralwindowinfo()
    ms.close()

    # average down to one channel per spw
    # output is in DATA column
    spw_list = []
    avg_list = []
    for k in spw_info.keys():
        spw_list.append( spw_info[k]['SpectralWindowId'] )
        avg_list.append( spw_info[k]['NumChan'])
        
    ms_avg = f'{wdir}/{msfile}.avg'
    if not os.path.exists(ms_avg):
        tb.open(msfile)
        datacol = 'CORRECTED'
        if 'CORRECTED' not in tb.colnames():
            datacol = 'DATA'
            
        logging.info(f'averaging {msfile}, using {datacol}')
        split(vis=msfile, outputvis=ms_avg, keepflags=False,
              spw=','.join([str(s) for s in spw_list]), width=avg_list,
              datacolumn=datacol)
        tb.close()

    # get spw and scan info on split file
    ms.open(ms_avg)
    scan_info = ms.getscansummary()
    ms.close()

    # process by scan
    scans_sorted = [int(s) for s in scan_info.keys()]
    scans_sorted.sort()
    for scan_no in scans_sorted:

        scan_str = f'scan_{int(scan_no):02d}'
        scan_no_dir = f'{scandir}/{scan_no}'
        scan_output =f'{scan_no_dir}/{scan_str}'
        if not os.path.exists(scan_no_dir):
            os.mkdir(scan_no_dir)
            
        npy_avg_scan = f'{scan_no_dir}/{scan_str}.npy'
        ms_avg_scan = f'{scan_no_dir}/{scan_str}.ms'
        
        if not os.path.exists(npy_avg_scan):
        
            logging.info(f'splitting scan {scan_no} from {ms_avg_scan}')
            if not os.path.exists(ms_avg_scan):
                split(vis=ms_avg, outputvis=ms_avg_scan, scan=scan_no, datacolumn='DATA')

            # get data from ms
            u, v, vis, wt, time = export_ms(ms_avg_scan, tb, ms)

            # field info to link things
            msmd.open(ms_avg_scan)
            field_id = msmd.fieldsforscan(int(scan_no))
            if len(field_id) > 1:
                logging.warning(f'{len(field_id)} fields for scan {s}')

            field_name = msmd.namesforfields(field_id)
            if len(field_name) > 1:
                logging.warning(f'{len(field_name)} fields for scan {s}')
            field_name = field_name[0]
            msmd.close()
            
            np.save(npy_avg_scan, np.array([u, v, vis, wt, time, field_name], dtype=object))

        else:
            logging.info(f'loading visibilites from {npy_avg_scan}')
            u, v, vis, wt, time, field_name = np.load(npy_avg_scan, allow_pickle=True)

        if not keep_scan_npy:
            os.unlink(npy_avg_scan)

        if not keep_scan_ms and os.path.exists(ms_avg_scan):
            shutil.rmtree(ms_avg_scan)

        if len(vis) == 0:
            logging.info(f'{scan_no}: no visibilities')
            continue

        # reweighting as suggested by Loomis+
        # unclear where factor 0.5 w.r.t. above comes from probably no of d.o.f
        # 2 dof would be: rew = 2*len(w) / np.sum( (Re**2.0 + Im**2.0) * w )
        wgt_mean = np.mean(wt)
        data_std = np.std(vis)
        rew = (1/data_std**2)/wgt_mean
        logging.info(f'reweighting value (1dof): {rew}')
        wt *= rew

        # check visiblity weights sensible
        # multiply by sqrt(2) assuming Re and Im independent
        fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True)
        _ = ax[0].hist(np.sqrt(2)*vis.real*np.sqrt(wt), bins=100, density=True, label='Real')
        _ = ax[1].hist(np.sqrt(2)*vis.imag*np.sqrt(wt), bins=100, density=True, label='Imag')
        x = np.linspace(-3,3)
        for a in ax:
            a.plot(x, np.max(_[0])*np.exp(-(x**2)/2))
            a.set_xlabel('snr per visibility')
            a.legend()
        ax[0].set_ylabel('density')
        fig.savefig(f'{scan_output}.vis_snr.png')
        plt.close(fig)

        # absolute sum of weighted visibilities
        # (point source variability with no location)
        # unique returns sorted times
        times = np.unique(time)

        # times in minutes
        tplot1 = (time-np.min(time))*24*60
        tplot2 = (times-np.min(times))*24*60
        Dt = np.max(times) - np.min(times)*24*60
        dt = np.median(np.diff(times))*24*60*60 # dt in seconds

        v_abs = []
        for t in times:
            ok = time == t
            v_abs.append( np.dot( np.abs(vis[ok]), np.sqrt(wt[ok])) )
            
        v_abs = np.array(v_abs)

        # plot output
        fig, ax = plt.subplots(2, sharex=True, figsize=(8,4))
        ax[0].plot(tplot1, vis.real, '.', label='Real', markersize=0.3)
        ax[1].plot(tplot2, np.abs(v_abs), '.', label='$|Sum(V.w)|$')
        ax[1].set_xlabel('Time / minutes')
        ax[0].set_ylabel('Real')
        ax[1].set_ylabel('$|Sum(V.w)|$')
        fig.tight_layout()
        fig.savefig(f'{scan_output}.vis_time.png')
        plt.close(fig)

        # smooth light curve
        nw = 60
        if len(times) < 60:
            nw = len(times)//2
        ws = np.arange(nw)+1
        T = np.zeros((nw,len(times)))
        wpk = []
        pk = []
        
        vmin = 1e5
        for i,wi in enumerate(ws):
            conv = np.convolve(v_abs, np.repeat(1,wi)/wi, mode='valid')
            conv = (conv-np.mean(conv))*np.sqrt(wi) + np.mean(conv)
            T[i, (wi-1)//2:len(v_abs)-wi//2] = conv

            # find significant +ve outliers
            clipped, _, _ = scipy.stats.sigmaclip(conv)
            mn, std = np.mean(clipped), np.std(clipped)
            ok = np.where(T[i] > mn + 3*std)[0]
            for o in ok:
                wpk.append(wi)
                pk.append(tplot2[o])
                
            if np.min(conv) < vmin:
                vmin = np.min(conv)
            
        wpk = np.array(wpk)
        pk = np.array(pk)

        fig, ax = plt.subplots(2, figsize=(8,6), sharex=True,
                               gridspec_kw={'height_ratios':[2,1]})
        ax[0].imshow(T, aspect='auto', origin='lower', vmin=vmin,
                     extent=(np.min(tplot2), np.max(tplot2),
                             np.min(ws)-0.5, np.max(ws)+0.5))
        ax[0].plot(pk, wpk, '+w')
        ax[1].plot(tplot2, np.abs(v_abs), '.', label='$|Sum(V.w)|$')
        ax[1].set_xlabel('Time / minutes')
        ax[1].set_ylabel('$|Sum(V.w)|$')
        ax[0].set_ylabel(f'smoothing width / $\Delta t$={dt:3.2f} seconds')
        fig.tight_layout()
        fig.savefig(f'{scan_output}.vis_time_smooth.png')
        plt.close(fig)
        
        field_dir = f'{wdir}/{field_name}'
        if not os.path.exists(field_dir):
            os.mkdir(field_dir)
            
        linked = f'{field_dir}/{scan_str}.vis_time_smooth.png'
        if not os.path.exists(linked):
            os.symlink(os.path.abspath(f'{scan_output}.vis_time_smooth.png'), linked)

    if not keep_avg_ms and os.path.exists(ms_avg):
        shutil.rmtree(ms_avg)


def uv_search(msfile, ra_off=0, dec_off=0, make_fits=False):

    flux = []
    time = []
    scan = []
    dt = 2 # seconds

    out = listobs(msfile)
    
    for k in out.keys():
        if 'scan' in k:
            t0 = out[k]['0']['BeginTime']
            t1 = out[k]['0']['EndTime']
            ts = np.arange(t0, t1, dt/(24*60*60))
            ts = np.append(ts, t1)
            
    wdir = os.path.dirname(msfile)
    cleandir = f'{wdir}/cleans'
    if not os.path.exists(cleandir):
        os.mkdir(cleandir)
            
    for i, t in enumerate(ts[:-1]):
                
        t0_ = mjd2date(t)
        t1_ = mjd2date(ts[i+1])
        t0_str = t0_.strftime('%H:%M:%S') + t0_.strftime(".%f")[:2]
        t1_str = t1_.strftime('%H:%M:%S') + t1_.strftime(".%f")[:2]
        tmid = (t + ts[i+1])/2
        time.append(tmid)
        
        uvmodelfit(vis=msfile, timerange=f"{t0_str}~{t1_str}",
                   comptype='P',
                   sourcepar=[1,ra_off,dec_off], varypar=[True,False,False],
                   outfile=f'{wdir}/tmp.cl')
        tb.open(f'{wdir}/tmp.cl')
        flux.append(abs(tb.getcol('Flux')[0][0]))
        tb.close()
        
        if make_fits:
            os.system(f'rm -rf {wdir}/tmpimage*')
            tclean(vis=f'{msfile}',imagename=f'{wdir}/tmpimage',
                   cell='0.5arcsec',
                   imsize=[256,256],interactive=False,niter=0,
                   timerange=f"{t0_str}~{t1_str}")
            exportfits(imagename=f'{wdir}/tmpimage.image',
                       fitsimage=f'{cleandir}/{i:04d}.fits')

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, flux, '.')
    fig.savefig(f'{wdir}/uv_flux_time.png')
    plt.close(fig)
    np.save(f'{wdir}/timeflux.npy', np.vstack((time,flux)))

    os.system(f'rm -rf {wdir}/tmp.cl')
    if make_fits:
        os.system(f'rm -rf {wdir}/tmpimage*')


#msfile = 'uid___A002_Xbd8c60_X181a.ms.split.cal'
#var_search(msfile)
