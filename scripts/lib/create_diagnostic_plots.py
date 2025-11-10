import arviz as az
from   matplotlib.backends.backend_pdf import PdfPages
#==============================================================================
def create_diagnostic_plots(pdf_filename,idata):

    # Filtering the list of parameters to plot
    vars_main = ["^(?!.*(model_spec|continuum|_z|v_rad|sigma)).*"]

    pdf_pages = PdfPages(pdf_filename)

    print(" - Chains")
    az.plot_trace(idata, var_names=vars_main, divergences=True, filter_vars="regex")
    pdf_pages.savefig()

    print(" - Pair plot")
    az.plot_pair(idata,var_names=vars_main,kind='kde',divergences=True, filter_vars="regex")
    pdf_pages.savefig()

    print(" - Autocorr plot")
    az.plot_autocorr(idata, var_names=vars_main, filter_vars="regex")
    pdf_pages.savefig()

    print(" - Energy plot")
    az.plot_energy(idata)
    pdf_pages.savefig()

    pdf_pages.close()   

    return
