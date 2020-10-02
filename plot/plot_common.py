import matplotlib.pyplot as plt

def save_figure(path_fig, fmt, transparent = False):

    if fmt == 'pdf':
        
        if transparent:

            print('Warning: cannot save transparent figure with fmt = pdf')

        plt.savefig(path_fig)

    elif fmt == 'png':

        plt.savefig(path_fig, dpi = 300, transparent = transparent)

    else:

        raise NotImplementedError('Can only save to format pdf or png, not {:}'.format(fmt))

    return
