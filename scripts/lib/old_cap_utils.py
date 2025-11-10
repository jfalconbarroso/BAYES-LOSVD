from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance, cKDTree
from scipy import ndimage
#==============================================================================
""""
#####################################################################

Copyright (C) 2001-2017, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

If you have found this software useful for your
research, we would appreciate an acknowledgment to use of
`the Voronoi binning method by Cappellari & Copin (2003)'.

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.

#####################################################################

NAME:
    VORONOI_2D_BINNING

AUTHOR:
      Michele Cappellari, University of Oxford
      michele.cappellari_at_physics.ox.ac.uk

PURPOSE:
      Perform adaptive spatial binning of Integral-Field Spectroscopic
      (IFS) data to reach a chosen constant signal-to-noise ratio per bin.
      This method is required for the proper analysis of IFS
      observations, but can also be used for standard photometric
      imagery or any other two-dimensional data.
      This program precisely implements the algorithm described in
      section 5.1 of the reference below.

EXPLANATION:
      Further information on VORONOI_2D_BINNING algorithm can be found in
      Cappellari M., Copin Y., 2003, MNRAS, 342, 345
      http://adsabs.harvard.edu/abs/2003MNRAS.342..345C

CALLING SEQUENCE:

        binNum, xBin, yBin, xBar, yBar, sn, nPixels, scale = \
            voronoi_2d_binning(x, y, signal, noise, targetSN,
                               cvt=True, pixelsize=None, plot=True,
                               quiet=True, sn_func=None, wvt=True)

INPUTS:
           X: Vector containing the X coordinate of the pixels to bin.
              Arbitrary units can be used (e.g. arcsec or pixels).
              In what follows the term "pixel" refers to a given
              spatial element of the dataset (sometimes called "spaxel" in
              the IFS community): it can be an actual pixel of a CCD
              image, or a spectrum position along the slit of a long-slit
              spectrograph or in the field of view of an IFS
              (e.g. a lenslet or a fiber).
              It is assumed here that pixels are arranged in a regular
              grid, so that the pixel size is a well defined quantity.
              The pixel grid however can contain holes (some pixels can be
              excluded from the binning) and can have an irregular boundary.
              See the above reference for an example and details.
           Y: Vector (same size as X) containing the Y coordinate
              of the pixels to bin.
      SIGNAL: Vector (same size as X) containing the signal
              associated with each pixel, having coordinates (X,Y).
              If the `pixels' are actually the apertures of an
              integral-field spectrograph, then the signal can be
              defined as the average flux in the spectral range under
              study, for each aperture.
              If pixels are the actual pixels of the CCD in a galaxy
              image, the signal will be simply the counts in each pixel.
       NOISE: Vector (same size as X) containing the corresponding
              noise (1 sigma error) associated with each pixel.
    TARGETSN: The desired signal-to-noise ratio in the final
              2D-binned data. E.g. a S/N~50 per pixel may be a
              reasonable value to extract stellar kinematics
              information from galaxy spectra.

KEYWORDS:
         CVT: Set this keyword to skip the Centroidal Voronoi Tessellation
              (CVT) step (vii) of the algorithm in Section 5.1 of
              Cappellari & Copin (2003).
              This may be useful if the noise is strongly non Poissonian,
              the pixels are not optimally weighted, and the CVT step
              appears to introduces significant gradients in the S/N.
              A similar alternative consists of using the /WVT keyword below.
        PLOT: Set this keyword to produce a plot of the two-dimensional
              bins and of the corresponding S/N at the end of the
              computation.
     PIXSIZE: Optional pixel scale of the input data.
              This can be the size of a pixel of an image or the size
              of a spaxel or lenslet in an integral-field spectrograph.
            - The value is computed automatically by the program, but
              this can take a long times when (X, Y) have many elements.
              In those cases the PIXSIZE keyword should be given.
     SN_FUNC: Generic function to calculate the S/N of a bin with spaxels
              "index" with the form: "sn = func(index, signal, noise)".
              If this keyword is not set, or is set to None, the program
              uses the _sn_func(), included in the program file, but
              another function can be adopted if needed.
              See the documentation of _sn_func() for more details.
       QUIET: by default the program shows the progress while accreting
              pixels and then while iterating the CVT. Set this keyword
              to avoid printing progress results.
         WVT: When this keyword is set, the routine bin2d_cvt_equal_mass is
              modified as proposed by Diehl & Statler (2006, MNRAS, 368, 497).
              In this case the final step of the algorithm, after the bin-accretion
              stage, is not a modified Centroidal Voronoi Tessellation, but it uses
              a Weighted Voronoi Tessellation.
              This may be useful if the noise is strongly non Poissonian,
              the pixels are not optimally weighted, and the CVT step
              appears to introduces significant gradients in the S/N.
              A similar alternative consists of using the /NO_CVT keyword above.
              If you use the /WVT keyword you should also include a reference to
              `the WVT modification proposed by Diehl & Statler (2006).'

OUTPUTS:
   BINNUMBER: Vector (same size as X) containing the bin number assigned
              to each input pixel. The index goes from zero to Nbins-1.
              IMPORTANT: THIS VECTOR ALONE IS ENOUGH TO MAKE *ANY* SUBSEQUENT
              COMPUTATION ON THE BINNED DATA. EVERYTHING ELSE IS OPTIONAL!

        XBIN: Vector (size Nbins) of the X coordinates of the bin generators.
              These generators uniquely define the Voronoi tessellation.
              Note: USAGE OF THIS VECTOR IS DEPRECATED AS IT CAN CAUSE CONFUSION
        YBIN: Vector (size Nbins) of Y coordinates of the bin generators.
              Note: USAGE OF THIS VECTOR IS DEPRECATED AS IT CAN CAUSE CONFUSION
        XBAR: Vector (size Nbins) of X coordinates of the bins luminosity
              weighted centroids. Useful for plotting interpolated data.
        YBAR: Vector (size Nbins) of Y coordinates of the bins luminosity
              weighted centroids.
          SN: Vector (size Nbins) with the final SN of each bin.
     NPIXELS: Vector (size Nbins) with the number of pixels of each bin.
       SCALE: Vector (size Nbins) with the scale length of the Weighted
              Voronoi Tessellation, when the /WVT keyword is set.
              In that case SCALE is *needed* together with the coordinates
              XBIN and YBIN of the generators, to compute the tessellation
              (but one can also simply use the BINNUMBER vector).

PROCEDURES USED:
      The following procedures are contained in the main VORONOI_2D_BINNING program.
          _SN_FUNC          -- Example routine to calculate the S/N of a bin.
          WEIGHTED_CENTROID -- computes weighted centroid of one bin
          BIN_ROUNDNESS     -- equation (5) of Cappellari & Copin (2003)
          BIN_ACCRETION     -- steps (i)-(v) in section 5.1
          REASSIGN_BAD_BINS -- steps (vi)-(vii) in section 5.1
          CVT_EQUAL_MASS    -- the modified Lloyd algorithm in section 4.1
          COMPUTE_USEFUL_BIN_QUANTITIES -- self explanatory
          DISPLAY_PIXELS    -- plotting of colored pixels

MODIFICATION HISTORY:
      V1.0.0: First implementation. Michele Cappellari, Leiden, June 2001
      V2.0.0: Major revisions. Stable version. MC, Leiden, 11 September 2001
      V2.1.0: First released version. Written documentation.
          MC, Vicenza, 13 February 2003
      V2.2.0: Added computation of useful bin quantities in output. Deleted some
          safety checks for zero size bins in CVT. Minor polishing of the code.
          MC, Leiden, 11 March 2003
      V2.3.0: Unified the three tests to stop the accretion of one bin.
          This can improve some bins at the border. MC, Leiden, 9 April 2003
      V2.3.1: Do *not* assume the first bin is made of one single pixel.
          Added computation of S/N scatter and plotting of 1-pixel bins.
          MC, Leiden, 13 April 2003
      V2.4.0: Addedd basic error checking of input S/N. Reintroduced the
          treatment for zero-size bins in CVT, which was deleted in V2.2.
          Thanks to Robert Sharp and Kambiz Fathi for reporting problems.
          MC, Leiden, 10 December 2003.
      V2.4.1: Added /QUIET keyword and verbose output during the computation.
          After suggestion by Richard McDermid. MC, Leiden, 14 December 2003
      V2.4.2: Use LONARR instead of INTARR to define the CLASS vector,
          to be able to deal with big images. Thanks to Tom Statler.
          MC, Leiden, 4 August 2004
      V2.4.3: Corrected bug introduced in version 2.3.1. It went undetected
          for a long time because it could only happen in special conditions.
          Now we recompute the index of the good bins after computing all
          centroids of the reassigned bins in reassign_bad_bins. Many thanks
          to Simona Ghizzardi for her clear analysis of the problem and
          the solution. MC, Leiden, 29 November 2004
      V2.4.4: Prevent division by zero for pixels with signal=0
          and noise=sqrt(signal)=0, as can happen from X-ray data.
          MC, Leiden, 30 November 2004
      V2.4.5: Added BIN2D prefix to internal routines to avoid possible
          naming conflicts. MC, Leiden, 3 December 2004
      V2.4.6: Added /NO_CVT keyword to optionally skip the CVT step of
          the algorithm. MC, Leiden, 27 August 2005
      V2.4.7: Verify that SIGNAL and NOISE are non negative vectors.
          MC, Leiden, 27 September 2005
      V2.4.8: Use geometric centroid of a bin during the bin-accretion stage,
          to allow the routine to deal with negative signal (e.g. in
          background-subtracted X-ray images). Thanks to Steven Diehl for
          pointing out the usefulness of dealing with negative signal.
          MC, Leiden, 23 December 2005
      V2.5.0: Added two new lines of code and the corresponding /WVT keyword
          to implement the nice modification to the algorithm proposed by
          Diehl & Statler (2006). MC, Leiden, 9 March 2006
      V2.5.1: Updated documentation. MC, Oxford, 3 November 2006
      V2.5.2: Print number of unbinned pixels. MC, Oxford, 28 March 2007
      V2.5.3: Fixed program stop, introduced in V2.5.0, with /NO_CVT keyword.
          MC, Oxford, 3 December 2007
      V2.5.4: Improved color shuffling for final plot.
          MC, Oxford, 30 November 2009
      V2.5.5: Added PIXSIZE keyword. MC, Oxford, 28 April 2010
      V2.5.6: Use IDL intrinsic function DISTANCE_MEASURE for
          automatic pixelSize, when PIXSIZE keyword is not given.
          MC, Oxford, 11 November 2011
      V2.5.7: Included safety termination criterion of Lloyd algorithm
          to prevent loops using /WVT. MC, Oxford, 24 March 2012
      V2.5.8: Update Voronoi tessellation at the exit of bin2d_cvt_equal_mass.
          This is only done when using /WVT, as DIFF may not be zero at the
          last iteration. MC, La Palma, 15 May 2012
      V2.6.0: Included new SN_FUNCTION to illustrate the fact that the user can
          define his own function to estimate the S/N of a bin if needed.
          MC, London, 19 March 2014
      V3.0.0: Translated from IDL into Python and tested against the original.
          MC, London, 19 March 2014
      V3.0.1: Support both Python 2.7 and Python 3. MC, Oxford, 25 May 2014
      V3.0.2: Avoid potential runtime warning while plotting.
          MC, Oxford, 2 October 2014
      V3.0.3: Use for loop to calculate Voronoi tessellation of large arrays
          to reduce memory usage. Thanks to Peter Weilbacher (Potsdam) for
          reporting the problem and providing the solution.
          MC, Oxford, 31 March 2016
      V3.0.4: Included keyword "sn_func" to pass a function which
          calculates the S/N of a bin, rather than editing _sn_func().
          Included test to prevent the addition of a pixel from
          ever decreasing the S/N during the accretion stage.
          MC, Oxford, 12 April 2016
      V3.0.5: Fixed deprecation warning in Numpy 1.11. MC, Oxford, 18 April 2016
      V3.0.6: Use interpolation='nearest' to avoid crash on MacOS.
          Thanks to Kyle Westfall (Portsmouth) for reporting the problem.
          Allow for zero noise. MC, Oxford, 14 June 2016
      V3.0.7: Print execution time. MC, Oxford, 23 January 2017
      V3.0.8: New voronoi_tessellation() function. MC, Oxford, 15 February 2017
      V3.0.9: Do not iterate down to diff==0 in _cvt_equal_mass().
          Request `pixelsize` when dataset is large. Thanks to Davor Krajnovic
          (Potsdam) for the feedback. Make `quiet` really quiet.
          Fixd some instances where sn_func() was not being used (only relevant
          when passing the `sn_func` keyword). MC, Oxford, 10 July 2017
      V3.1.0: Use cKDTree for un-weighted Voronoi Tessellation.
          Removed loop over bins from Lloyd's algorithm with CVT.
          MC, Oxford, 17 July 2017

"""

#----------------------------------------------------------------------------

def _sn_func(index, signal=None, noise=None):
    """
    Default function to calculate the S/N of a bin with spaxels "index".

    The Voronoi binning algorithm does not require this function to have a
    specific form and this default one can be changed by the user if needed
    by passing a different function as

        ... = voronoi_2d_binning(..., sn_func=sn_func)

    The S/N returned by sn_func() does not need to be an analytic
    function of S and N.

    There is also no need for sn_func() to return the actual S/N.
    Instead sn_func() could return any quantity the user needs to equalize.

    For example sn_func() could be a procedure which uses ppxf to measure
    the velocity dispersion from the coadded spectrum of spaxels "index"
    and returns the relative error in the dispersion.

    Of course an analytic approximation of S/N, like the one below,
    speeds up the calculation.

    :param index: integer vector of length N containing the indices of
        the spaxels for which the combined S/N has to be returned.
        The indices refer to elements of the vectors signal and noise.
    :param signal: vector of length M>N with the signal of all spaxels.
    :param noise: vector of length M>N with the noise of all spaxels.
    :return: scalar S/N or another quantity that needs to be equalized.
    """

    sn = np.sum(signal[index])/np.sqrt(np.sum(noise[index]**2))

    # The following commented line illustrates, as an example, how one
    # would include the effect of spatial covariance using the empirical
    # Eq.(1) from http://adsabs.harvard.edu/abs/2015A%26A...576A.135G
    # Note however that the formula is not accurate for large bins.
    #
    # sn /= 1 + 1.07*np.log10(index.size)

    return  sn

#----------------------------------------------------------------------

def voronoi_tessellation(x, y, xnode, ynode, scale):
    """
    Computes (Weighted) Voronoi Tessellation of the pixels grid

    """
    if scale[0] == 1:  # non-weighted VT
        tree = cKDTree(np.column_stack([xnode, ynode]))
        classe = tree.query(np.column_stack([x, y]))[1]
    else:
        if x.size < 1e4:
            classe = np.argmin(((x[:, None] - xnode)**2 + (y[:, None] - ynode)**2)/scale**2, axis=1)
        else:  # use for loop to reduce memory usage
            classe = np.zeros(x.size, dtype=int)
            for j, (xj, yj) in enumerate(zip(x, y)):
                classe[j] = np.argmin(((xj - xnode)**2 + (yj - ynode)**2)/scale**2)

    return classe

#----------------------------------------------------------------------

def _centroid(x, y, density):
    """
    Computes weighted centroid of one bin.
    Equation (4) of Cappellari & Copin (2003)

    """
    mass = np.sum(density)
    xBar = x.dot(density)/mass
    yBar = y.dot(density)/mass

    return xBar, yBar

#----------------------------------------------------------------------

def _roundness(x, y, pixelSize):
    """
    Implements equation (5) of Cappellari & Copin (2003)

    """
    n = x.size
    equivalentRadius = np.sqrt(n/np.pi)*pixelSize
    xBar, yBar = np.mean(x), np.mean(y)  # Geometric centroid here!
    maxDistance = np.sqrt(np.max((x - xBar)**2 + (y - yBar)**2))
    roundness = maxDistance/equivalentRadius - 1.

    return roundness

#----------------------------------------------------------------------

def _accretion(x, y, signal, noise, targetSN, pixelsize, quiet, sn_func):
    """
    Implements steps (i)-(v) in section 5.1 of Cappellari & Copin (2003)

    """
    n = x.size
    classe = np.zeros(n, dtype=int)  # will contain the bin number of each given pixel
    good = np.zeros(n, dtype=bool)   # will contain 1 if the bin has been accepted as good

    # For each point, find the distance to all other points and select the minimum.
    # This is a robust but slow way of determining the pixel size of unbinned data.
    #
    if pixelsize is None:
        if x.size < 1e4:
            pixelsize = np.min(distance.pdist(np.column_stack([x, y])))
        else:
            raise ValueError("Dataset is large: Provide `pixelsize`")

    currentBin = np.argmax(signal/noise)  # Start from the pixel with highest S/N
    SN = sn_func(currentBin, signal, noise)

    # Rough estimate of the expected final bins number.
    # This value is only used to give an idea of the expected
    # remaining computation time when binning very big dataset.
    #
    w = signal/noise < targetSN
    maxnum = int(np.sum((signal[w]/noise[w])**2)/targetSN**2 + np.sum(~w))

    # The first bin will be assigned CLASS = 1
    # With N pixels there will be at most N bins
    #
    for ind in range(1, n+1):

        if not quiet:
            print(ind, ' / ', maxnum)

        classe[currentBin] = ind  # Here currentBin is still made of one pixel
        xBar, yBar = x[currentBin], y[currentBin]    # Centroid of one pixels

        while True:

            if np.all(classe):
                break  # Stops if all pixels are binned

            # Find the unbinned pixel closest to the centroid of the current bin
            #
            unBinned = np.flatnonzero(classe == 0)
            k = np.argmin((x[unBinned] - xBar)**2 + (y[unBinned] - yBar)**2)

            # (1) Find the distance from the closest pixel to the current bin
            #
            minDist = np.min((x[currentBin] - x[unBinned[k]])**2 + (y[currentBin] - y[unBinned[k]])**2)

            # (2) Estimate the `roundness' of the POSSIBLE new bin
            #
            nextBin = np.append(currentBin, unBinned[k])
            roundness = _roundness(x[nextBin], y[nextBin], pixelsize)

            # (3) Compute the S/N one would obtain by adding
            # the CANDIDATE pixel to the current bin
            #
            SNOld = SN
            SN = sn_func(nextBin, signal, noise)

            # Test whether (1) the CANDIDATE pixel is connected to the
            # current bin, (2) whether the POSSIBLE new bin is round enough
            # and (3) whether the resulting S/N would get closer to targetSN
            #
            if (np.sqrt(minDist) > 1.2*pixelsize or roundness > 0.3
                or abs(SN - targetSN) > abs(SNOld - targetSN) or SNOld > SN):
                if SNOld > 0.8*targetSN:
                    good[currentBin] = 1
                break

            # If all the above 3 tests are negative then accept the CANDIDATE
            # pixel, add it to the current bin, and continue accreting pixels
            #
            classe[unBinned[k]] = ind
            currentBin = nextBin

            # Update the centroid of the current bin
            #
            xBar, yBar = np.mean(x[currentBin]), np.mean(y[currentBin])

        # Get the centroid of all the binned pixels
        #
        binned = classe > 0
        if np.all(binned):
            break  # Stop if all pixels are binned
        xBar, yBar = np.mean(x[binned]), np.mean(y[binned])

        # Find the closest unbinned pixel to the centroid of all
        # the binned pixels, and start a new bin from that pixel.
        #
        unBinned = np.flatnonzero(classe == 0)
        k = np.argmin((x[unBinned] - xBar)**2 + (y[unBinned] - yBar)**2)
        currentBin = unBinned[k]    # The bin is initially made of one pixel
        SN = sn_func(currentBin, signal, noise)

    classe *= good  # Set to zero all bins that did not reach the target S/N

    return classe, pixelsize

#----------------------------------------------------------------------------

def _reassign_bad_bins(classe, x, y):
    """
    Implements steps (vi)-(vii) in section 5.1 of Cappellari & Copin (2003)

    """
    # Find the centroid of all successful bins.
    # CLASS = 0 are unbinned pixels which are excluded.
    #
    good = np.unique(classe[classe > 0])
    xnode = ndimage.mean(x, labels=classe, index=good)
    ynode = ndimage.mean(y, labels=classe, index=good)

    # Reassign pixels of bins with S/N < targetSN
    # to the closest centroid of a good bin
    #
    bad = classe == 0
    index = voronoi_tessellation(x[bad], y[bad], xnode, ynode, [1])
    classe[bad] = good[index]

    # Recompute all centroids of the reassigned bins.
    # These will be used as starting points for the CVT.
    #
    good = np.unique(classe)
    xnode = ndimage.mean(x, labels=classe, index=good)
    ynode = ndimage.mean(y, labels=classe, index=good)

    return xnode, ynode

#----------------------------------------------------------------------------

def _cvt_equal_mass(x, y, signal, noise, xnode, ynode, pixelsize, quiet, sn_func, wvt):
    """
    Implements the modified Lloyd algorithm
    in section 4.1 of Cappellari & Copin (2003).

    NB: When the keyword WVT is set this routine includes
    the modification proposed by Diehl & Statler (2006).

    """
    dens2 = (signal/noise)**4     # See beginning of section 4.1 of CC03
    scale = np.ones_like(xnode)   # Start with the same scale length for all bins

    for it in range(1, xnode.size):  # Do at most xnode.size iterations

        xnode_old, ynode_old = xnode.copy(), ynode.copy()
        classe = voronoi_tessellation(x, y, xnode, ynode, scale)

        # Computes centroids of the bins, weighted by dens**2.
        # Exponent 2 on the density produces equal-mass Voronoi bins.
        # The geometric centroids are computed if WVT keyword is set.
        #
        good = np.unique(classe)
        if wvt:
            for k in good:
                index = np.flatnonzero(classe == k)   # Find subscripts of pixels in bin k.
                xnode[k], ynode[k] = np.mean(x[index]), np.mean(y[index])
                sn = sn_func(index, signal, noise)
                scale[k] = np.sqrt(index.size/sn)  # Eq. (4) of Diehl & Statler (2006)
        else:
            mass = ndimage.sum(dens2, labels=classe, index=good)
            xnode = ndimage.sum(x*dens2, labels=classe, index=good)/mass
            ynode = ndimage.sum(y*dens2, labels=classe, index=good)/mass

        diff2 = np.sum((xnode - xnode_old)**2 + (ynode - ynode_old)**2)
        diff = np.sqrt(diff2)/pixelsize

        if not quiet:
            print('Iter: %4i  Diff: %.4g' % (it, diff))

        if diff < 0.1:
            break

    # If coordinates have changed, re-compute (Weighted) Voronoi Tessellation of the pixels grid
    #
    if diff > 0:
        classe = voronoi_tessellation(x, y, xnode, ynode, scale)
        good = np.unique(classe)  # Check for zero-size Voronoi bins

    # Only return the generators and scales of the nonzero Voronoi bins

    return xnode[good], ynode[good], scale[good], it

#-----------------------------------------------------------------------

def _compute_useful_bin_quantities(x, y, signal, noise, xnode, ynode, scale, sn_func):
    """
    Recomputes (Weighted) Voronoi Tessellation of the pixels grid to make sure
    that the class number corresponds to the proper Voronoi generator.
    This is done to take into account possible zero-size Voronoi bins
    in output from the previous CVT (or WVT).

    """
    # classe will contain the bin number of each given pixel
    classe = voronoi_tessellation(x, y, xnode, ynode, scale)

    # At the end of the computation evaluate the bin luminosity-weighted
    # centroids (xbar, ybar) and the corresponding final S/N of each bin.
    #
    xbar = np.empty_like(xnode)
    ybar = np.empty_like(xnode)
    sn = np.empty_like(xnode)
    area = np.empty_like(xnode)
    good = np.unique(classe)
    for k in good:
        index = np.flatnonzero(classe == k)   # index of pixels in bin k.
        xbar[k], ybar[k] = _centroid(x[index], y[index], signal[index])
        sn[k] = sn_func(index, signal, noise)
        area[k] = index.size

    return classe, xbar, ybar, sn, area

#-----------------------------------------------------------------------

def _display_pixels(x, y, counts, pixelsize):
    """
    Display pixels at coordinates (x, y) coloured with "counts".
    This routine is fast but not fully general as it assumes the spaxels
    are on a regular grid. This needs not be the case for Voronoi binning.

    """
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int(round((xmax - xmin)/pixelsize) + 1)
    ny = int(round((ymax - ymin)/pixelsize) + 1)
    img = np.full((nx, ny), np.nan)  # use nan for missing data
    j = np.round((x - xmin)/pixelsize).astype(int)
    k = np.round((y - ymin)/pixelsize).astype(int)
    img[j, k] = counts

    plt.imshow(np.rot90(img), interpolation='nearest', cmap='prism',
               extent=[xmin - pixelsize/2, xmax + pixelsize/2,
                       ymin - pixelsize/2, ymax + pixelsize/2])

#----------------------------------------------------------------------

def voronoi_2d_binning(x, y, signal, noise, targetSN, cvt=True,
                         pixelsize=None, plot=True, quiet=True,
                         sn_func=None, wvt=True):
    """
    PURPOSE:
          Perform adaptive spatial binning of Integral-Field Spectroscopic
          (IFS) data to reach a chosen constant signal-to-noise ratio per bin.
          This method is required for the proper analysis of IFS
          observations, but can also be used for standard photometric
          imagery or any other two-dimensional data.
          This program precisely implements the algorithm described in
          section 5.1 of the reference below.

    EXPLANATION:
          Further information on VORONOI_2D_BINNING algorithm can be found in
          Cappellari M., Copin Y., 2003, MNRAS, 342, 345

    CALLING SEQUENCE:

        binNum, xBin, yBin, xBar, yBar, sn, nPixels, scale = \
            voronoi_2d_binning(x, y, signal, noise, targetSN,
                               cvt=True, pixelsize=None, plot=True,
                               quiet=True, sn_func=None, wvt=True)

    """
    # This is the main program that has to be called from external programs.
    # It simply calls in sequence the different steps of the algorithms
    # and optionally plots the results at the end of the calculation.
    assert x.size == y.size == signal.size == noise.size, \
        'Input vectors (x, y, signal, noise) must have the same size'
    assert np.all((noise > 0) & np.isfinite(noise)), \
        'NOISE must be positive and finite'

    if sn_func is None:
        sn_func = _sn_func

    # Perform basic tests to catch common input errors
    #
    if sn_func(noise > 0, signal, noise) < targetSN:
        raise ValueError("""Not enough S/N in the whole set of pixels.
            Many pixels may have noise but virtually no signal.
            They should not be included in the set to bin,
            or the pixels should be optimally weighted.
            See Cappellari & Copin (2003, Sec.2.1) and README file.""")
    if np.min(signal/noise) > targetSN:
        raise ValueError('All pixels have enough S/N and binning is not needed')

    if not quiet:
        print('Bin-accretion...')
    classe, pixelsize = _accretion(
        x, y, signal, noise, targetSN, pixelsize, quiet, sn_func)
    if not quiet:
        print(np.max(classe), ' initial bins.')
        print('Reassign bad bins...')
    xnode, ynode = _reassign_bad_bins(classe, x, y)
    if not quiet:
        print(xnode.size, ' good bins.')
    if cvt:
        if not quiet:
            print('Modified Lloyd algorithm...')
        xnode, ynode, scale, it = _cvt_equal_mass(
            x, y, signal, noise, xnode, ynode, pixelsize, quiet, sn_func, wvt)
        if not quiet:
            print(it - 1, ' iterations.')
    else:
        scale = np.ones_like(xnode)
    classe, xBar, yBar, sn, area = _compute_useful_bin_quantities(
        x, y, signal, noise, xnode, ynode, scale, sn_func)
    w = area == 1
    if not quiet:
        print('Unbinned pixels: ', np.sum(w), ' / ', x.size)
        print('Fractional S/N scatter (%):', np.std(sn[~w] - targetSN, ddof=1)/targetSN*100)

    if plot:
        plt.clf()
        plt.subplot(211)
        rnd = np.argsort(np.random.random(xnode.size))  # Randomize bin colors
        _display_pixels(x, y, rnd[classe], pixelsize)
        plt.plot(xnode, ynode, '+w', scalex=False, scaley=False) # do not rescale after imshow()
        plt.xlabel('R (arcsec)')
        plt.ylabel('R (arcsec)')
        plt.title('Map of Voronoi bins')

        plt.subplot(212)
        rad = np.sqrt(xBar**2 + yBar**2)  # Use centroids, NOT generators
        plt.plot(rad[~w], sn[~w], 'or', label='Voronoi bins')
        plt.xlabel('R (arcsec)')
        plt.ylabel('Bin S/N')
        plt.axis([np.min(rad), np.max(rad), 0, np.max(sn)])  # x0, x1, y0, y1
        if np.sum(w) > 0:
            plt.plot(rad[w], sn[w], 'xb', label='single spaxels')
        plt.axhline(targetSN)
        plt.legend()
        plt.pause(1)  # allow plot to appear in certain cases

    return classe, xnode, ynode, xBar, yBar, sn, area, scale

#==============================================================================
#
# NAME:
#   LOG_REBIN
#
# MODIFICATION HISTORY:
#   V1.0.0: Using interpolation. Michele Cappellari, Leiden, 22 October 2001
#   V2.0.0: Analytic flux conservation. MC, Potsdam, 15 June 2003
#   V2.1.0: Allow a velocity scale to be specified by the user.
#       MC, Leiden, 2 August 2003
#   V2.2.0: Output the optional logarithmically spaced wavelength at the
#       geometric mean of the wavelength at the border of each pixel.
#       Thanks to Jesus Falcon-Barroso. MC, Leiden, 5 November 2003
#   V2.2.1: Verify that lamRange[0] < lamRange[1].
#       MC, Vicenza, 29 December 2004
#   V2.2.2: Modified the documentation after feedback from James Price.
#       MC, Oxford, 21 October 2010
#   V2.3.0: By default now preserve the shape of the spectrum, not the
#       total flux. This seems what most users expect from the procedure.
#       Set the keyword /FLUX to preserve flux like in previous version.
#       MC, Oxford, 30 November 2011
#   V3.0.0: Translated from IDL into Python. MC, Santiago, 23 November 2013
#   V3.1.0: Fully vectorized log_rebin. Typical speed up by two orders of magnitude.
#       MC, Oxford, 4 March 2014
#   V3.1.1: Updated documentation. MC, Oxford, 16 August 2016

def log_rebin(lamRange, spec, oversample=False, velscale=None, flux=False):
    """
    Logarithmically rebin a spectrum, while rigorously conserving the flux.
    Basically the photons in the spectrum are simply redistributed according
    to a new grid of pixels, with non-uniform size in the spectral direction.

    When the flux keyword is set, this program performs an exact integration
    of the original spectrum, assumed to be a step function within the
    linearly-spaced pixels, onto the new logarithmically-spaced pixels.
    The output was tested to agree with the analytic solution.

    :param lamRange: two elements vector containing the central wavelength
        of the first and last pixels in the spectrum, which is assumed
        to have constant wavelength scale! E.g. from the values in the
        standard FITS keywords: LAMRANGE = CRVAL1 + [0,CDELT1*(NAXIS1-1)].
        It must be LAMRANGE[0] < LAMRANGE[1].
    :param spec: input spectrum.
    :param oversample: Oversampling can be done, not to loose spectral resolution,
        especally for extended wavelength ranges and to avoid aliasing.
        Default: OVERSAMPLE=1 ==> Same number of output pixels as input.
    :param velscale: velocity scale in km/s per pixels. If this variable is
        not defined, then it will contain in output the velocity scale.
        If this variable is defined by the user it will be used
        to set the output number of pixels and wavelength scale.
    :param flux: (boolean) True to preserve total flux. In this case the
        log rebinning changes the pixels flux in proportion to their
        dLam so the following command will show large differences
        beween the spectral shape before and after LOG_REBIN:

           plt.plot(exp(logLam), specNew)  # Plot log-rebinned spectrum
           plt.plot(np.linspace(lamRange[0], lamRange[1], spec.size), spec)

        By defaul, when this is False, the above two lines produce
        two spectra that almost perfectly overlap each other.
    :return: [specNew, logLam, velscale]

    """
    lamRange = np.asarray(lamRange)
    if len(lamRange) != 2:
        raise ValueError('lamRange must contain two elements')
    if lamRange[0] >= lamRange[1]:
        raise ValueError('It must be lamRange[0] < lamRange[1]')
    s = spec.shape
    if len(s) != 1:
        raise ValueError('input spectrum must be a vector')
    n = s[0]
    if oversample:
        m = int(n*oversample)
    else:
        m = int(n)

    dLam = np.diff(lamRange)/(n - 1.)        # Assume constant dLam
    lim = lamRange/dLam + [-0.5, 0.5]        # All in units of dLam
    borders = np.linspace(*lim, num=n+1)     # Linearly
    logLim = np.log(lim)

    c = 299792.458                           # Speed of light in km/s
    if velscale is None:                     # Velocity scale is set by user
        velscale = np.diff(logLim)/m*c       # Only for output
    else:
        logScale = velscale/c
        m = int(np.diff(logLim)/logScale)    # Number of output pixels
        logLim[1] = logLim[0] + m*logScale

    newBorders = np.exp(np.linspace(*logLim, num=m+1)) # Logarithmically
    k = (newBorders - lim[0]).clip(0, n-1).astype(int)

    specNew = np.add.reduceat(spec, k)[:-1]  # Do analytic integral
    specNew *= np.diff(k) > 0    # fix for design flaw of reduceat()
    specNew += np.diff((newBorders - borders[k])*spec[k])

    if not flux:
        specNew /= np.diff(newBorders)

    # Output log(wavelength): log of geometric mean
    logLam = np.log(np.sqrt(newBorders[1:]*newBorders[:-1])*dLam)

    return specNew, logLam, velscale
#==============================================================================
###############################################################################
#
# NAME:
#   DETERMINE_GOODPIXELS
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Leiden, 9 September 2005
#   V1.0.1: Made a separate routine and included additional common emission lines.
#       MC, Oxford 12 January 2012
#   V2.0.0: Translated from IDL into Python. MC, Oxford, 10 December 2013
#   V2.0.1: Updated line list. MC, Oxford, 8 January 2014
#   V2.0.2: Use redshift instead of velocity as input for higher accuracy at large z.
#       MC, Lexington, 31 March 2015
#   V2.0.3: Includes `width` keyword after suggestion by George Privon (Univ. Florida).
#       MC, Oxford, 2 July 2018

def determine_goodpixels(logLam, lamRangeTemp, z, width=800.):
    """
    Generates a list of goodpixels to mask a given set of gas emission
    lines. This is meant to be used as input for PPXF.

    :param logLam: Natural logarithm np.log(wave) of the wavelength in
        Angstrom of each pixel of the log rebinned *galaxy* spectrum.
    :param lamRangeTemp: Two elements vectors [lamMin2, lamMax2] with the minimum
        and maximum wavelength in Angstrom in the stellar *template* used in PPXF.
    :param z: Estimate of the galaxy redshift.
    :return: vector of goodPixels to be used as input for pPXF

    """
#                     -----[OII]-----    Hdelta   Hgamma   Hbeta   -----[OIII]-----  -[NI]-   [OI]    -----[NII]-----   Halpha   -----[SII]-----
    lines = np.array([3726.03, 3728.82, 4101.76, 4340.47, 4861.33, 4958.92, 5006.84, 5200.0,  6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
    dv = np.full_like(lines, width)  # width/2 of masked gas emission region in km/s
    c = 299792.458 # speed of light in km/s

    flag = False
    for line, dvj in zip(lines, dv):
        flag |= (np.exp(logLam) > line*(1 + z)*(1 - dvj/c)) \
              & (np.exp(logLam) < line*(1 + z)*(1 + dvj/c))

    flag |= np.exp(logLam) > lamRangeTemp[1]*(1 + z)*(1 - 900.0/c)   # Mask edges of
    flag |= np.exp(logLam) < lamRangeTemp[0]*(1 + z)*(1 + 900.0/c)   # stellar library

    return np.flatnonzero(~flag)

#==============================================================================
###############################################################################
# NAME:
#   GAUSSIAN_FILTER1D
#
# MODIFICATION HISTORY:
#   V1.0.0: Written as a replacement for the Scipy routine with the same name,
#       to be used with variable sigma per pixel. MC, Oxford, 10 October 2015

def gaussian_filter1d(spec, sig):
    """
    Convolve a spectrum by a Gaussian with different sigma for every pixel.
    If all sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating a template library for SDSS data, this implementation
    is 60x faster than a naive for loop over pixels.

    :param spec: vector with the spectrum to convolve
    :param sig: vector of sigma values (in pixels) for every pixel
    :return: spec convolved with a Gaussian with dispersion sig

    """
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

    conv_spectrum = np.sum(a*gau, 0)

    return conv_spectrum
#==============================================================================
"""
    Copyright (C) 2016, Michele Cappellari
    E-mail: michele.cappellari_at_physics.ox.ac.uk

    Updated versions of the software are available from my web page
    http://purl.org/cappellari/software

    See example at the bottom for usage instructions.

    V1.0.0: Created to emulate my IDL procedure with the same name.
        Michele Cappellari, Oxford, 28 March 2014
    V1.0.1: Fixed treatment of optional parameters. MC, Oxford, 6 June 2014
    V1.0.2: Avoid potential runtime warning. MC, Oxford, 2 October 2014
    V1.0.3: Return axis. MC, Oxford, 26 March 2015
    V1.0.4: Return image instead of axis. MC, Oxford, 15 July 2015
    V1.0.5: Removes white gaps from rotated images using edgecolors.
        MC, Oxford, 5 October 2015
    V1.0.6: Pass kwargs to graphics functions.
        MC, Campos do Jordao, Brazil, 23 November 2015
    V1.0.7: Check that input (x,y) come from an axis-aligned image.
        MC, Oxford, 28 January 2016
    V1.0.8: Fixed deprecation warning in Numpy 1.11. MC, Oxford, 22 April 2016
    V1.1.0: Fixed program stop with kwargs. Included `colorbar` keyword.
        MC, Oxford, 18 May 2016

"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from scipy.spatial import distance
import numpy as np

#from sauron_colormap import sauron
##############################################################################

def display_pixels(x, y, val, pixelsize=None, vmin=None, vmax=None,
                   angle=None, colorbar=False, label=None, nticks=7, cmap=None, **kwargs):
    """
    Display vectors of square pixels at coordinates (x,y) coloured with "val".
    An optional rotation around the origin can be applied to the whole image.

    The pixels are assumed to be taken from a regular cartesian grid with
    constant spacing (like an axis-aligned image), but not all elements of
    the grid are required (missing data are OK).

    This routine is designed to be fast even with large images and to produce
    minimal file sizes when the output is saved in a vector format like PDF.

    """
    x, y, val = map(np.ravel, [x, y, val])

    if not (x.size == y.size == val.size):
        raise ValueError('The vectors (x, y, val) must have the same size')

    if vmin is None:
        vmin = np.min(val)

    if vmax is None:
        vmax = np.max(val)

    if pixelsize is None:
        pixelsize = np.min(distance.pdist(np.column_stack([x, y])))

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    x1 = (x - xmin)/pixelsize
    y1 = (y - ymin)/pixelsize
    nx = int(round((xmax - xmin)/pixelsize) + 1)
    ny = int(round((ymax - ymin)/pixelsize) + 1)
    mask = np.ones((nx, ny), dtype=bool)
    img = np.empty((nx, ny))
    j = np.round(x1).astype(int)
    k = np.round(y1).astype(int)
    mask[j, k] = 0
    img[j, k] = val
    img = np.ma.masked_array(img, mask)

    if np.any(np.abs(np.append(j - x1, k - y1)) > 0.1):
        raise ValueError('The coordinates (x, y) must come from an axis-aligned image')

    ax = plt.gca()

    if (angle is None) or (angle == 0):
        
        if cmap == None:
           cmap=kwargs.get("cmap", sauron)

        img = ax.imshow(np.rot90(img), interpolation='none',
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[xmin-pixelsize/2, xmax+pixelsize/2,
                                ymin-pixelsize/2, ymax+pixelsize/2])

    else:

        x, y = np.ogrid[xmin-pixelsize/2 : xmax+pixelsize/2 : (nx+1)*1j,
                        ymin-pixelsize/2 : ymax+pixelsize/2 : (ny+1)*1j]
        ang = np.radians(angle)
        x, y = x*np.cos(ang) - y*np.sin(ang), x*np.sin(ang) + y*np.cos(ang)

        mask1 = np.ones_like(x, dtype=bool)
        mask1[:-1, :-1] *= mask  # Flag the four corners of the mesh
        mask1[:-1, 1:] *= mask
        mask1[1:, :-1] *= mask
        mask1[1:, 1:] *= mask
        x = np.ma.masked_array(x, mask1)  # Mask is used for proper plot range
        y = np.ma.masked_array(y, mask1)

        img = ax.pcolormesh(x, y, img, cmap=kwargs.get("cmap", sauron),
                            vmin=vmin, vmax=vmax, edgecolors="face")
        ax.axis('image')

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ticks = MaxNLocator(nticks).tick_values(vmin, vmax)
        cbar = plt.colorbar(img, cax=cax, ticks=ticks)
        cbar.solids.set_edgecolor("face")  # Remove gaps in PDF http://stackoverflow.com/a/15021541
        if label:
            cbar.set_label(label)
        plt.sca(ax)  # Activate main plot before returning


    ax.minorticks_on()
    ax.tick_params(length=10, width=1, which='major')
    ax.tick_params(length=5, width=1, which='minor')

    return img
#==============================================================================
"""
NAME:
    display_bins(x, y, binNum, velBin)

AUTHOR:
    Michele Cappellari, University of Oxford
    E-mail: michele.cappellari_at_physics.ox.ac.uk

PURPOSE:
    This simple routine illustrates how to display a Voronoi binned map.

INPUTS:
    (x, y): (length npix) Coordinates of the original spaxels before binning;
    binNum: (length npix) Bin number corresponding to each (x, y) pair,
            as provided in output by the voronoi_2d_binning() routine;
    velBin: (length nbins) Quantity associated to each bin, resulting
            e.g. from the kinematic extraction from the binned spectra.

MODIFICATION HISTORY:
    V1.0.0: Michele Cappellari, Oxford, 15 January 2015
    V1.0.1: Further input checks. MC, Oxford, 15 July 2015

"""

import numpy as np

#from cap_display_pixels import display_pixels

def display_bins(x, y, binNum, velBin, cmap=None):

    if not (x.size == y.size == binNum.size):
        raise ValueError('The vectors (x, y, binNum) must have the same size')

    if np.unique(binNum).size != velBin.size:
        raise ValueError('velBin size does not match number of bins')

    img = display_pixels(x, y, velBin[binNum], cmap=cmap)

    return img
#==============================================================================
"""
  Copyright (C) 2014-2015, Michele Cappellari
  E-mail: michele.cappellari_at_physics.ox.ac.uk
  http://purl.org/cappellari

  V1.0: Created to emulate my IDL procedure with the same name.
        Michele Cappellari, Oxford, 28 March 2014
  V1.1: Included reversed colormap. MC, Oxford, 9 August 2015

"""
from matplotlib import colors

##############################################################################

# V1.0: SAURON colormap by Michele Cappellari & Eric Emsellem, Leiden, 10 July 2001
#
# Start with these 7 equally spaced coordinates, then add 4 additional points
# x = findgen(7)*255/6. + 1
# 1.0  43.5  86.0  128.5  171.0  213.5  256.0
#
# x = [1.0, 43.5, 86.0, 86.0+20, 128.5-10, 128.5, 128.5+10, 171.0-20, 171.0, 213.5, 256.0]
# red =   [0.0, 0.0, 0.4,  0.5, 0.3, 0.0, 0.7, 1.0, 1.0,  1.0, 0.9]
# green = [0.0, 0.0, 0.85, 1.0, 1.0, 0.9, 1.0, 1.0, 0.85, 0.0, 0.9]
# blue =  [0.0, 1.0, 1.0,  1.0, 0.7, 0.0, 0.0, 0.0, 0.0,  0.0, 0.9]

_cdict = {'red':[(0.000,   0.01,   0.01),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.4,    0.4),
                 (0.414,   0.5,    0.5),
                 (0.463,   0.3,    0.3),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.7,    0.7),
                 (0.590,   1.0,    1.0),
                 (0.668,   1.0,    1.0),
                 (0.834,   1.0,    1.0),
                 (1.000,   0.9,    0.9)],
        'green':[(0.000,   0.01,   0.01),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.85,   0.85),
                 (0.414,   1.0,    1.0),
                 (0.463,   1.0,    1.0),
                 (0.502,   0.9,    0.9),
                 (0.541,   1.0,    1.0),
                 (0.590,   1.0,    1.0),
                 (0.668,   0.85,   0.85),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.9,    0.9)],
         'blue':[(0.000,   0.01,   0.01),
                 (0.170,   1.0,    1.0),
                 (0.336,   1.0,    1.0),
                 (0.414,   1.0,    1.0),
                 (0.463,   0.7,    0.7),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.0,    0.0),
                 (0.590,   0.0,    0.0),
                 (0.668,   0.0,    0.0),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.9,    0.9)]
         }

_rdict = {'red':[(0.000,   0.9,    0.9),
                 (0.170,   1.0,    1.0),
                 (0.336,   1.0,    1.0),
                 (0.414,   1.0,    1.0),
                 (0.463,   0.7,    0.7),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.3,    0.3),
                 (0.590,   0.5,    0.5),
                 (0.668,   0.4,    0.4),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.01,   0.01)],
        'green':[(0.000,   0.9,    0.9),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.85,   0.85),
                 (0.414,   1.0,    1.0),
                 (0.463,   1.0,    1.0),
                 (0.502,   0.9,    0.9),
                 (0.541,   1.0,    1.0),
                 (0.590,   1.0,    1.0),
                 (0.668,   0.85,   0.85),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.01,   0.01)],
         'blue':[(0.000,   0.9,    0.9),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.0,    0.0),
                 (0.414,   0.0,    0.0),
                 (0.463,   0.0,    0.0),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.7,    0.7),
                 (0.590,   1.0,    1.0),
                 (0.668,   1.0,    1.0),
                 (0.834,   1.0,    1.0),
                 (1.000,   0.01,   0.01)]
         }

sauron = colors.LinearSegmentedColormap('sauron', _cdict)
sauron_r = colors.LinearSegmentedColormap('sauron_r', _rdict)

#==============================================================================
#==============================================================================
#==============================================================================
