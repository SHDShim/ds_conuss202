# -*- coding: utf-8 -*-
"""
version 2018.06.21
Last updated on 2018/06/19
Created on Sun Aug 10 14:36:33 2014
@author: DanShim
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from time import ctime
import glob
import time
import datetime
import fileinput


def get_datafile_name(where_to_find='in_kfit'):
    """
    Internal function to obtain datafile name from in_kfit
    This looks for the following line in in_kfit
    `(2) exp. data file  ::   DAC4_FeOOH_15GPa_7095s.dat10 2column`
    """
    with open(where_to_find, "r") as f:
        for l in f:
            result = l.find('exp. data file')
            if result != -1:
                words = l.split('::')
                input_filen = ((words[1].lstrip()).split(" "))[0]
                return input_filen
    return None


def read_datafile():
    """
    Internal function to read data file
    """
    filen = get_datafile_name()
    x = []
    y = []
    with open(filen, "r") as f:
        for l in f:
            result = l.find('Mask')
            if result == -1:
                xy = ((l.lstrip()).rstrip()).split(" ")
                x.append(float(xy[0]))
                y.append(float(xy[1]))
    return np.asarray(x), np.asarray(y)


def read_mask():
    """
    Internal function to read mask from data file
    """
    filen = get_datafile_name()
    mask_str = []
    with open(filen, "r") as f:
        for l in f:
            result = l.find('Mask')
            if result != -1:
                mask_str.append(l.rstrip())
    return mask_str


def get_input_file(where_to_find='in_kref'):
    """
    Internal function input file name from in_kref
    """
    with open(where_to_find, "r") as f:
        for l in f:
            result = l.find('material data input file')
            if result != -1:
                words = l.split('::')
                input_filen = (words[1].replace(" ", "")).rstrip()
                return input_filen
    return None


def get_initial_parameters():
    """
    Internal function to find input parameters to vary
    """
    input_filen = get_input_file()
    param_str = []
    with open(input_filen, "r") as f:
        for l in f:
            if l[0] == "%":
                param_str.append(' '.join(l.rstrip().split()))
    return param_str


def print_initial_parameters():
    """
    Internal function to print initial parameters from input file
    """
    param_str = get_initial_parameters()
    for s in param_str:
        print(s)


def print_fit_parameters():
    """
    Internal function to print fit parameters for fit process
    """
    param_str = get_fit_parameters()
    for p in param_str:
        print(p)


def get_fit_parameters():
    """
    Internal function to print fitting results
    """
    output_filen = 'out_kctl'
    read_on = False
    param_str = []
    with open(output_filen, "r") as f:
        for l in f:
            if l.find('Results') != -1:
                read_on = True
            if read_on and (l.find('@') != -1):
                parameter, initial_value, final_value = get_from_kctl(l)
                param_str.append(parameter +
                                 ":= {0:.7e} is changed by {1:.4f} to:".
                                 format(initial_value,
                                        (final_value/initial_value)-1.))
                param_str.append(parameter + " := {:.7e}".format(final_value))
    return param_str


def get_chisq_kctl():
    """
    Obtain chisq from kctl
    """
    output_filen = 'out_kctl'
    read_on = False
    with open(output_filen, "r") as f:
        for l in f:
            if l.find('Results') != -1:
                read_on = True
            if read_on and (l.find('Chi**2 (normalized)') != -1):
                num_str = ((((l.rstrip()).split(':'))[1]).split('+-'))[0]
                chisq = float(num_str.replace("D", "E"))
                return str(chisq)
    return None


def get_chisq_kmco():
    """
    Obtain chisq from kmco
    """
    output_filen = 'out_kmco'
    read_on = False
    with open(output_filen, "r") as f:
        for l in f:
            if read_on:
                num_str = ((l.rstrip().lstrip()).split(' '))[0]
                chisq = num_str  # I do not convert because of ********
                return chisq
            if l.find('Best sampling:') != -1:
                read_on = True
    return None


def get_mco_parameters():
    """
    Internal function to get mco input parameters
    """
    file_in = get_input_file()
    output_filen = file_in + '.mco'
    param_str = []
    with open(output_filen, "r") as f:
        for l in f:
            if l[0] == "%":
                param_str.append(" ".join(l.rstrip().split()))
    return param_str


def print_mco_parameters():
    """
    Internal function to print mco initial value and search results
    """
    print("** Initial values **")
    print_initial_parameters()
    print("** Final values **")
    param_str = get_mco_parameters()
    for p in param_str:
        print(p)


def get_from_kctl(line):
    """
    Internal function to find initial and final value from `out_kctl`
    It reads line like:
    ` | @  is2 :=  5.1253E-01 |  1 |  5.3574E-01 +- 3.5E-02 |   6.594 |`
    """
    parameter = ((((line.rstrip()).split('|'))[1].split(":="))[0]).lstrip()
    num_str = \
        (((((line.rstrip()).split('|')[1]).split(":=")[1]).
          lstrip()).split(" "))[0]
    initial_value = float(num_str.replace("D", "E"))
    num_str = ((line.rstrip()).split('|')[3]).split("+-")[0]
    final_value = float(num_str.replace("D", "E"))
    return ' '.join(parameter.split()), initial_value, final_value


def readcxp(filen='data_graph.cxp'):
    """
    Internal function to read data points included in the fit
    """
    data = np.loadtxt(filen)
    x, y, errorbar = data.T
    return x, y, errorbar


def readfit(filen='data_graph.fit'):
    """
    Internal function to read fit curve
    """
    fit_data = np.loadtxt(filen)
    xfit, yfit = fit_data.T
    return xfit, yfit


def readrsd(rsd='data_graph.rsd'):
    """
    Internal function to read fit residue
    """
    data = np.loadtxt(rsd)
    x, y, errorbar = data.T
    return x, y, errorbar


def readdata():
    """
    Internal function to read all necessary fitting components in once
    """
    x, y, errorbar = readcxp()
    xfit, yfit = readfit()
    return x, y, errorbar, xfit, yfit


def guess(plot_data=True):
    """
    Generate calculated spectrum for a given set of parameters
    """
    os.system('conuss rmfx')
    plot(plot_data=plot_data)


def search(alarm=True, plot_result=True, print_result=True, log=True,
           line_to_range=None):
    """
    Conduct Monte-Carlo search for the parameteral space given by a user

    alarm = setup alarm at the finished
    plot_result = plot search result at the end
    print_result = print search result at the end
    """
    os.system('conuss mco')
    if alarm:
        os.system('say "It is finished."')
    print('==============================================')
    print('==========CONUSS MCO SEARCH FINISHED==========')
    print('==============================================')
    if print_result:
        print_mco_parameters()
    plot_search(show_plot=plot_result)
    if log:
        write_log(command='search', line_to_range=line_to_range)


def fit(alarm=True, print_result=True, plot_result=True, log=True):
    """
    Calculate Mossbauer spectrum for given parameters

    alarm = setup alarm at the finished
    plot_result = plot search result at the end
    print_result = print search result at the end
    """
    os.system('conuss rmfx')
    x_initial, y_initial = readfit()
    # Conduct fit
    os.system('conuss fit tmp')
    if alarm:
        os.system('say "It is finished."')
    print('==============================================')
    print('==============CONUSS FIT FINISHED=============')
    print('==============================================')
    if print_result:
        print_fit_parameters()
    if plot_result:
        plot(x_initial=x_initial, y_initial=y_initial)
    if log:
        write_log(command='fit')


def normlogspecfile(cxp='data_graph.cxp', fit='data_graph.fit'):
    """
    Internal function, currently unused
    """
    xobs, yobs, yerr = readcxp(filen=cxp)
    xfit, yfit = readfit(filen=fit)

    # normalize for plotting in linear scale and staking
    ymax = (np.log10(yobs)).max()
    ymin = (np.log10(yobs)).min()
    ylogobs = (np.log10(yobs) - ymin)/(ymax-ymin)
    ylogerr = (np.log10(yerr/yobs + 1.))/(ymax-ymin)
    ylogfit = (np.log10(yfit) - ymin)/(ymax-ymin)

    return xobs, ylogobs, ylogerr, xfit, ylogfit


def normlogsynspecfile(fit='data_graph.fit'):
    """
    Internal function, currently unused
    """
    xfit, yfit = readfit(filen=fit)

    # normalize for plotting in linear scale and staking
    ymax = (np.log10(yfit)).max()
    ymin = (np.log10(yfit)).min()
    ylog = (np.log10(yfit) - ymin)/(ymax-ymin)

    return xfit, ylog


def plot(time_lim=(0., 140.), y_lim=None, x_initial=None, y_initial=None,
         plot_data=True, show_plot=True):
    """
    Plot the results
    """
    x, y, errorbar, xfit, yfit = readdata()
    x_rsd, y_rsd, errorbar_rsd = readrsd()
    x_raw, y_raw = read_datafile()

    f = plt.figure()
    plt.subplots_adjust(hspace=0.001)

    ax2 = plt.subplot2grid((7, 1), (6, 0), rowspan=1)
    ax2.set_xlim(time_lim)
    ax2.errorbar(x_rsd, y_rsd, color='k', ms=2, fmt='o', mfc='white',
                 capsize=0)
    ax2.set_xlabel('Time (ns)')
    ax2.axhline(0., color='k')

    ax1 = plt.subplot2grid((7, 1), (0, 0), rowspan=6, sharex=ax2)

    if plot_data:
        ax1.plot(x_raw, y_raw, 'k.', label='Data excluded')
        ax1.errorbar(x, y, yerr=errorbar, color='k',
                     fmt='o', mfc='white', capsize=0, label='Data included')
    ax1.plot(xfit, yfit, color='r', linewidth=2, label='Final')
    if x_initial is not None:
        ax1.plot(x_initial, y_initial, color='b', label='Initial')
    ax1.set_ylabel('Counts (log)')
    if y_lim is None:
        ax1.set_ylim([y.min()*0.8, y.max()*1.2])
    else:
        ax1.set_ylim(y_lim)
    ax1.set_yscale('log')
    ax1.set_xlim(time_lim)  # don't know why but required line
    cwd = os.getcwd().split('/')
    ax1.set_title(cwd[-2]+', '+cwd[-1]+', '+ctime())
    ax1.legend()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.savefig('fit_result.pdf')
    if show_plot:
        plt.show()


def plot_search(show_plot=True):
    """
    Plot the research results
    """
    os.system('conuss rmfx')
    x_initial, y_initial = readfit()
    # print result
    print_mco_parameters()
    # make temporary backup for the input file
    InFileName = get_input_file()
    os.rename(InFileName, InFileName+'.bak')
    # calculate for the best in mco file
    shutil.copy2(InFileName+'.mco', InFileName)
    os.system('conuss rmfx')
    plot(x_initial=x_initial, y_initial=y_initial, show_plot=show_plot)
    # put back the original input file
    os.remove(InFileName)
    os.rename(InFileName+'.bak', InFileName)


def insert_lines_log(lines, logfilen='conuss_log.txt'):
    """
    lines need to be a list of strings but without \n
    """
    backup_folder_name = "backups"
    if not os.path.isdir(backup_folder_name):
        os.makedirs(backup_folder_name)
    logfile = os.path.join(backup_folder_name, logfilen)
    with open(logfile, 'a+') as f:
        for l in lines:
            f.write(l+'\n')


def write_log(logfilen='conuss_log.txt', command='fit', verbose=False,
              line_to_range=None):
    """
    Internal function to write log
    """
    # get chisq
    if command == 'fit':
        chisq = get_chisq_kctl()
    elif command == 'search':
        chisq = get_chisq_kmco()
    if chisq is None:
        print('[error] No chisq available.  This run will not be logged.')
        return
    if verbose:
        print('** Logged information **')
        print('Chisq: ' + chisq + '\n')
    # get mask information
    mask_str = read_mask()
    if verbose:
        print('Masks')
        for m in mask_str:
            print(m+'\n')
    # check if backups directory exist
    backup_folder_name = "backups"
    if not os.path.isdir(backup_folder_name):
        os.makedirs(backup_folder_name)
    logfile = os.path.join(backup_folder_name, logfilen)
    # get time
    timestamp = datetime.datetime.now()
    timestamp_name = time.strftime("%Y%m%d_%H%M%S")
    fig_filen = time.strftime("%Y%m%d_%H%M%S")+'.pdf'
    bak_folder = os.path.join(backup_folder_name, timestamp_name)
    fig_file = os.path.join(backup_folder_name, fig_filen)
    if verbose:
        print('Backup directory: ', bak_folder)
    # get varied parameters and fit result
    if command == 'fit':
        param_str = get_fit_parameters()
    elif command == 'search':
        param_str = []
        param_str.append(
            "[Warning] This version does not log thickness change for search.")
        param_str.append("** Initial values **")
        p_str_i = get_initial_parameters()
        param_str += p_str_i
        param_str.append("** Final values **")
        p_str_f = get_mco_parameters()
        param_str += p_str_f
    # get filelist to zip
    os.mkdir(bak_folder)
    file_list = glob.glob("*")
    for f in file_list:
        if os.path.isfile(f):
            shutil.copy2(f, os.path.join(bak_folder, f))
    # rename figure.pdf and save in backup folder
    shutil.copy2('fit_result.pdf', fig_file)
    # append new information to log
    # if os.path.isfile(logfilen):
    with open(logfile, 'a+') as f:
        f.write('********************************************\n')
        f.write(command+' executed at: ' + str(timestamp)+'\n')
        if line_to_range is not None:
            f.write('Range Search for: \n' + line_to_range + '\n')
        for m in mask_str:
            f.write(m+'\n')
        f.write('Chisq = ' + str(chisq)+'\n')
        f.write('\n')
        for p in param_str:
            f.write(p+'\n')
        f.write('\n')
        f.write('Figure saved as: ' + fig_file+'\n')
        f.write('\n')


def make_infile(line_to_change, new_value, step, lineno=1):
    """
    Internal function to make an input file with a new value and a new step
    """
    param_str = line_to_change.split(':=')[0]
    new_line = param_str + \
        " := {0:.7e} {1:.7e} ::<-".format(new_value, step)
    input_filen = get_input_file()
    print('=================================================================')
    print('Ranged line: ', new_line)
    print('=================================================================')
    line_count = 0
    with fileinput.input(input_filen, inplace=True) as f:
        for line in f:
            if line.find("::<-") != -1:
                line_count += 1
                if line_count == lineno:
                    print(new_line.lstrip().rstrip())
                else:
                    print(line.lstrip().rstrip())
            else:
                print(line.lstrip().rstrip())


def get_line_to_range():
    """
    Internal function to get line information noted by `::<-`
    in the input file.
    """
    input_filen = get_input_file()
    lines = []
    with open(input_filen+'.org', "r") as f:
        for l in f:
            if l.find('::<-') != -1:
                lines.append(' '.join(l.split()))
    return lines


def search_range(v_0, v_f, n_pnts):
    """
    Conduct search for a range of values for one parameter noted by
    `::<-` in the input file.
    For example, to try from 360. to 0. for `theta1` variable,
    1. Open your input and go to the line for theta1 and then
    `% @ theta1 := 10. 5.` to `% @ theta1 := 10. 5. ::<-`
    2. Run `search_range(360., 0., 30)`

    v_0 = starting values
    v_f = stop values
    n_pnts = number of points
    """
    input_filen = get_input_file()
    shutil.copy2(input_filen, input_filen+'.org')
    line_to_range = get_line_to_range()[0]
    # print('Line for ranging is: ', line_to_range)
    if len(line_to_range) == 0:
        print('[Error] No line to range search')
        return
    step = (np.abs(v_f - v_0) / (n_pnts - 1.)) / 2.

    insert_lines_log([" ",
                      "======================================================",
                      "***Range Search begins***",
                      line_to_range,
                      "start = {0:.7e}, end = {1:.7e}, step = {2:.7e}, n_pnts = {3:.0f}".
                      format(v_0, v_f, step*2., n_pnts), " "])

    i = 1
    for v in np.linspace(v_0, v_f, n_pnts):
        make_infile(line_to_range, v, step)
        text = line_to_range + \
            '\n at {0:.7e} {1:.7e} \n {2:.0f} out of {3:.0f} iteration'.\
            format(v, step, i, n_pnts)
        search(alarm=False, plot_result=False, line_to_range=text)
        i += 1

    insert_lines_log([" ", "***Range Search ends***", line_to_range,
                      "start = {0:.7e}, end = {1:.7e}, step = {2:.7e}, n_pnts = {3:.0f}".
                      format(v_0, v_f, step, n_pnts),
                      "======================================================",
                      " "])

    return


def search_grid(param1, param2):
    """
    Conduct search for a range of values for one parameter noted by
    `::<-` in the input file.
    For example, to try from 360. to 0. for `theta1` variable,
    1. Open your input and go to the line for theta1 and then
    `% @ theta1 := 10. 5.` to `% @ theta1 := 10. 5. ::<-`
    2. Run `search_range(360., 0., 30)`

    param1, param2 = [v_0, v_f, n_pnts]
    v_0 = starting values
    v_f = stop values
    n_pnts = number of points
    """
    input_filen = get_input_file()
    shutil.copy2(input_filen, input_filen+'.org')
    line_to_range = get_line_to_range()
    # print('Line for ranging is: ', line_to_range)
    if len(line_to_range) != 2:
        print('[Error] There should be two lines for grid.')
        return

    step1 = (np.abs(param1[1] - param1[0]) / (param1[2] - 1.)) / 2.
    step2 = (np.abs(param2[1] - param2[0]) / (param2[2] - 1.)) / 2.

    insert_lines_log([" ",
                      "======================================================",
                      "***Grid Search begins***",
                      line_to_range[0],
                      "start = {0:.7e}, end = {1:.7e}, step = {2:.7e}, n_pnts = {3:.0f}".
                      format(param1[0], param1[1], step1*2., param1[2]),
                      line_to_range[1],
                      "start = {0:.7e}, end = {1:.7e}, step = {2:.7e}, n_pnts = {3:.0f}".
                      format(param2[0], param2[1], step2*2., param2[2]), " "
                      ])

    i = 1
    for v1 in np.linspace(param1[0], param1[1], param1[2]):
        for v2 in np.linspace(param2[0], param2[1], param2[2]):
            make_infile(line_to_range[0], v1, step1)
            make_infile(line_to_range[1], v2, step2, lineno=2)
            text = line_to_range[0] + line_to_range[1] + \
                '\n at {0:.7e} {1:.7e} \n at {2:.7e} {3:.7e} \n {4:.0f} out of {5:.0f} iteration'.\
                format(v1, step1, v2, step2, i, param1[2]*param2[2])
            search(alarm=False, plot_result=False, line_to_range=text)
            i += 1

    insert_lines_log([" ", "***Grid Search ends***", line_to_range[0],
                      "start = {0:.7e}, end = {1:.7e}, step = {2:.7e}, n_pnts = {3:.0f}".
                      format(param1[0], param1[1], step1*2., param1[2]),
                      line_to_range[1],
                      "start = {0:.7e}, end = {1:.7e}, step = {2:.7e}, n_pnts = {3:.0f}".
                      format(param2[0], param2[1], step2*2., param2[2]),
                      "======================================================",
                      " "])

    return
